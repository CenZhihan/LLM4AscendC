# gen_aligned_data.py
import numpy as np
import sys

def generate_aligned(shape, dtype, output_path):
    """
    生成32字节对齐的测试数据

    shape: tuple, 数据形状
    dtype: str, 数据类型 (fp16, fp32, int8)
    output_path: str, 输出文件路径
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    # 检查并调整对齐
    element_size = np.dtype(np_type).itemsize
    aligned_size = 32 // element_size

    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    # 生成随机数据
    data = np.random.rand(*adjusted_shape).astype(np_type)

    # 保存
    np.save(output_path, data)
    print(f"生成数据: {output_path}")
    print(f"  原始形状: {shape}")
    print(f"  调整形状: {tuple(adjusted_shape)}")
    print(f"  数据类型: {dtype}")
    print(f"  数据范围: [{data.min():.6f}, {data.max():.6f}]")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python3 gen_aligned_data.py <M> <N> <K> <dtype>")
        sys.exit(1)

    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    dtype = sys.argv[4]

    generate_aligned((M, N, K), dtype, f"input_{M}_{N}_{K}_{dtype}.npy")
