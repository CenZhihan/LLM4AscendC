import os
import json
import numpy as np
from typing import Optional
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 导入华为 NPU 支持
try:
    import torch_npu
    print("[INFO] torch_npu imported successfully")
except ImportError:
    print("[WARN] torch_npu not available, NPU support disabled")


class EmbeddingRetriever:
    """基于 sentence-transformers 的代码检索器"""

    def __init__(self, index_path: str, model_name: str = "BAAI/bge-small-zh-v1.5", devices: Optional[list[str]] = None):
        """
        初始化检索器

        Args:
            index_path: 索引存储路径
            model_name: sentence-transformers 模型名称
            devices: 设备列表，如 ['npu:0', 'npu:1']，None 表示自动检测所有可用 NPU
        """
        self.index_path = index_path
        self.model_name = model_name
        self.devices = devices or self._auto_detect_devices()
        self.model: Optional[SentenceTransformer] = None
        self.index: dict = {
            'embeddings': None,  # np.array
            'chunks': [],        # list[dict]: {'file': str, 'code': str, 'meta': dict}
            'model_name': model_name
        }

    def _auto_detect_devices(self) -> list[str]:
        """自动检测可用的 NPU 设备，不可用时回退到 CPU"""
        # 检测 NPU (华为昇腾)
        try:
            import torch_npu
            print(f"[DEBUG] torch_npu imported, torch.npu.is_available()={torch.npu.is_available()}")
            print(f"[DEBUG] torch.npu.device_count()={torch.npu.device_count()}")
            if torch.npu.is_available():
                num_npus = torch.npu.device_count()
                if num_npus > 0:
                    devices = [f'npu:{i}' for i in range(num_npus)]
                    print(f"[INFO] Auto-detected {num_npus} NPU(s): {devices}")
                    return devices
                else:
                    print("[WARN] torch.npu.is_available()=True but device_count=0")
            else:
                print("[WARN] torch.npu.is_available()=False")
        except ImportError as e:
            print(f"[WARN] torch_npu not installed: {e}")
        except Exception as e:
            print(f"[WARN] NPU detection error: {e}")

        print("[INFO] No NPU detected, using CPU")
        return ['cpu']

    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            # self.model = SentenceTransformer(self.model_name)
            # 使用华为 NPU 加载模型，不可用时回退到 CPU
            device = self.devices[0] if self.devices else 'cpu'
            print(f"[INFO] Loading model on {device}...")
            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"[INFO] Model loaded on {device}")
        return self.model

    def _encode_multi_npu(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        NPU 多卡并行编码

        Args:
            texts: 文本列表
            batch_size: 每个设备的批大小

        Returns:
            embeddings 数组
        """
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        num_devices = len(self.devices)
        chunk_size = (len(texts) + num_devices - 1) // num_devices

        # 分配数据到各个设备
        text_chunks = []
        for i in range(num_devices):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(texts))
            text_chunks.append(texts[start:end])

        # 创建共享队列收集结果
        manager = mp.Manager()
        result_queue = manager.Queue()

        # 启动进程
        processes = []
        for i, (device, texts_subset) in enumerate(zip(self.devices, text_chunks)):
            if not texts_subset:
                continue
            p = mp.Process(
                target=self._encode_on_device,
                args=(self.model_name, device, texts_subset, batch_size, result_queue, i)
            )
            p.start()
            processes.append(p)

        # 收集结果
        results = {}
        for _ in range(len(processes)):
            rank, embeddings = result_queue.get()
            results[rank] = embeddings

        # 等待所有进程结束
        for p in processes:
            p.join()

        # 按顺序合并结果
        all_embeddings = []
        for i in range(num_devices):
            if i in results:
                all_embeddings.append(results[i])

        return np.vstack(all_embeddings)

    @staticmethod
    def _encode_on_device(model_name: str, device: str, texts: list[str],
                          batch_size: int, result_queue, rank: int):
        """
        在指定设备上编码文本（用于多进程）
        """
        try:
            # 导入 torch_npu 以支持 NPU 设备
            try:
                import torch_npu
            except ImportError:
                pass

            # model = SentenceTransformer(model_name)
            # model.to(device)
            # 使用华为 NPU 设备，不可用时回退到 CPU
            print(f"[INFO] Loading model on {device} for rank {rank}...")
            model = SentenceTransformer(model_name, device=device)
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size
            )
            result_queue.put((rank, embeddings))
        except Exception as e:
            print(f"[ERROR] Device {device} encoding failed for rank {rank}: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put((rank, np.zeros((len(texts), 768))))  # 默认维度

    def build_index(self, chunks: list[dict], batch_size: int = 8, save_every: int = 20000) -> None:
        """
        构建向量索引（支持分批编码以降低内存峰值）

        Args:
            chunks: 代码片段列表，每个元素为 {'file': str, 'code': str, 'meta': dict}
            batch_size: 编码批大小（默认 8，降低内存使用）
            save_every: 每处理多少个 chunk 保存一次中间结果（用于断点续传）
        """
        if not chunks:
            print("[WARN] No chunks to index")
            return

        self._load_model()

        # 生成 embeddings
        texts = [chunk['code'] for chunk in chunks]
        total_chunks = len(texts)
        print(f"[INFO] Encoding {total_chunks} code chunks...")
        print(f"[DEBUG] Using devices: {self.devices}, batch_size: {batch_size}")
        import time
        import gc
        start_time = time.time()

        device = self.devices[0]
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 使用华为 NPU 设备，不可用时回退到 CPU
        print(f"[INFO] Using device: {device}")
        self.model.to(device)

        # 分批编码以降低内存峰值
        all_embeddings = []
        chunk_size = save_every  # 每批处理的 chunk 数量

        for batch_idx in range(0, total_chunks, chunk_size):
            batch_end = min(batch_idx + chunk_size, total_chunks)
            batch_texts = texts[batch_idx:batch_end]
            print(f"[INFO] Encoding batch {batch_idx//chunk_size + 1}/{(total_chunks + chunk_size - 1)//chunk_size} ({batch_idx}-{batch_end})...")

            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=batch_size
            )
            all_embeddings.append(batch_embeddings)

            # 释放内存
            del batch_embeddings
            del batch_texts
            gc.collect()

        # 合并所有 embeddings
        print("[INFO] Merging embeddings...")
        embeddings = np.vstack(all_embeddings)
        del all_embeddings
        gc.collect()

        elapsed = time.time() - start_time
        print(f"[DEBUG] Encoding completed in {elapsed:.1f} seconds ({total_chunks/elapsed:.1f} chunks/sec)")

        # 存储索引
        self.index['embeddings'] = embeddings
        self.index['chunks'] = chunks
        self.index['model_name'] = self.model_name

        print(f"[INFO] Index built: {len(chunks)} chunks, embedding dim: {embeddings.shape[1]}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索相似代码片段

        Args:
            query: 查询文本
            top_k: 返回top k个结果

        Returns:
            检索结果列表，每个元素为 {'file': str, 'code': str, 'score': float}
        """
        if self.index['embeddings'] is None or not self.index['chunks']:
            print("[WARN] Index not built or empty")
            return []

        self._load_model()

        # 编码查询
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 使用华为 NPU 设备，不可用时回退到 CPU
        device = self.devices[0] if self.devices else 'cpu'
        self.model.to(device)
        print(f"[INFO] Query encoding on device: {device}")
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # 计算余弦相似度
        embeddings = self.index['embeddings']
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 获取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.index['chunks'][idx]
            results.append({
                'file': chunk['file'],
                'code': chunk['code'],
                'score': float(similarities[idx]),
                'meta': chunk.get('meta', {})
            })

        return results

    def save_index(self) -> None:
        """保存索引到磁盘（index_path 作为目录）"""
        os.makedirs(self.index_path, exist_ok=True)

        # 保存 embeddings 为 numpy 文件
        embeddings_path = os.path.join(self.index_path, "embeddings.npy")
        if self.index['embeddings'] is not None:
            np.save(embeddings_path, self.index['embeddings'])

        # 保存 chunks 和元信息为 JSON
        meta_path = os.path.join(self.index_path, "meta.json")
        meta = {
            'model_name': self.model_name,
            'chunks': self.index['chunks'],
            'num_chunks': len(self.index['chunks'])
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Index saved to {self.index_path}/")

    def load_index(self) -> bool:
        """
        加载索引（index_path 作为目录，读取 embeddings.npy 和 meta.json）

        Returns:
            是否成功加载
        """
        embeddings_path = os.path.join(self.index_path, "embeddings.npy")
        meta_path = os.path.join(self.index_path, "meta.json")

        # 兼容旧格式（index 作为前缀）
        old_embeddings_path = f"{self.index_path}_embeddings.npy"
        old_meta_path = f"{self.index_path}_meta.json"

        # 优先尝试新格式，回退到旧格式
        if os.path.exists(embeddings_path) and os.path.exists(meta_path):
            pass  # 新格式存在
        elif os.path.exists(old_embeddings_path) and os.path.exists(old_meta_path):
            embeddings_path = old_embeddings_path
            meta_path = old_meta_path
        else:
            print(f"[INFO] Index files not found at {self.index_path}")
            return False

        try:
            # 加载 embeddings
            self.index['embeddings'] = np.load(embeddings_path)

            # 加载 chunks 和元信息
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            self.index['chunks'] = meta['chunks']
            self.index['model_name'] = meta.get('model_name', self.model_name)
            self.model_name = self.index['model_name']

            print(f"[INFO] Index loaded: {len(self.index['chunks'])} chunks")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load index: {e}")
            return False