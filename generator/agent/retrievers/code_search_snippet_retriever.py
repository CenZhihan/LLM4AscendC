"""Structural block retrieval over curated asc-devkit examples via manifest.

Uses a three-branch scoring model (metadata 0.45, BM25 0.30, dense 0.15, prior 0.10)
with hard context_type filtering and curation_tier gating.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from generator.config import (
    agent_code_search_snippet_knowledge_root,
    agent_code_search_snippet_max_chars,
    agent_code_search_snippet_top_k,
    rag_embedding_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_SNIPPET_CHARS = 80
_MAX_SNIPPET_CHARS = 8000
_BM25_K1 = 1.5
_BM25_B = 0.75
_DENSE_BATCH_SIZE = 8
_DENSE_TEXT_MAX_CHARS = 6000
_DENSE_TEXT_HEAD_CHARS = 4000
_DENSE_TEXT_TAIL_CHARS = 1500
_ENABLE_DENSE_ENV = "AGENT_CODE_SEARCH_SNIPPET_ENABLE_DENSE"

_WEIGHT_METADATA = 0.45
_WEIGHT_BM25 = 0.30
_WEIGHT_DENSE = 0.15
_WEIGHT_PRIOR = 0.10

_CANDIDATE_PREFILTER_MIN = 24
_CANDIDATE_PREFILTER_MAX = 64
_CANDIDATE_PREFILTER_MULTIPLIER = 24
_CANDIDATE_PREFILTER_STRICT_SCORE = 4.5
_CANDIDATE_PREFILTER_RELAXED_SCORE = 1.5

_TOKEN_NOISE_WORDS = frozenset({
    "agreement", "basis", "brief", "cann", "copyright", "detail", "file", "free",
    "huawei", "license", "ltd", "open", "program", "provided", "software",
    "technologies", "this", "warranties",
})
_API_SYMBOL_RE = re.compile(
    r"(?:AscendC::)?(?:[A-Z][A-Za-z0-9_]*|__[a-z0-9_]+__)(?:::[A-Za-z_][A-Za-z0-9_]*)*"
)
_API_SYMBOL_STOPWORDS = frozenset({
    "Agreement", "Basis", "Brief", "CANN", "Copyright", "Detail", "File", "Free",
    "Huawei", "License", "Ltd", "Open", "Program", "Provided", "Software",
    "Technologies", "This", "Warranties",
})

_CURATION_TIER_PRIORITY = {
    "preferred": 3.0,
    "default": 2.0,
    "fallback_only": 1.0,
    "hidden": 0.0,
}

_CONTEXT_TYPE_ORDER = [
    "kernel_src",
    "host_tiling_src",
    "host_infer_shape",
    "host_infer_dtype",
    "host_op_registration",
    "tiling_src",
    "project_json_src",
    "python_bind_src",
]

_OPERATOR_FAMILY_HINTS = {
    "activation": ("activation", "relu", "gelu", "softmax", "leakyrelu"),
    "convolution": ("conv", "convolution", "pointwise", "depthwise"),
    "matrix": ("matmul", "gemm", "mma", "matrix"),
    "normalization": ("layernorm", "rmsnorm", "normalization"),
    "elementwise": ("elementwise", "broadcast", "add", "mul", "div", "sub"),
    "reduce": ("reduce", "reducesum", "reducemax", "reducemin"),
    "pooling": ("pool", "pooling"),
}

_API_PATTERN_HINTS = {
    "tiling_storage_shape": ("getinputshape", "getoriginshape", "getshape", "shapesize"),
    "tiling_origin_shape_size": ("getoriginshape", "getshapesize", "shapesize"),
    "infer_shape_direct_dims": ("getdim", "setdim", "getdimnum", "setdimnum"),
    "infer_shape_copy_input_to_output": ("*yshape = *xshape", "yshape = xshape", "copy"),
    "infer_shape_set_output_dims": ("setdim", "setdimnum"),
    "infer_dtype_copy_input_dtype": ("getinputdatatype", "setoutputdatatype", "copy"),
    "get_attr_pointer": ("getattrpointer", "getattrs", "getattr"),
    "set_block_dim": ("setblockdim", "getblocknum"),
    "workspace_zero": ("getworkspacesizes", "workspace"),
    "opdef_register": ("op_add", "opdef", "registerop", "input(", "output("),
    "queue_api": ("tpipe", "tque<", "alloctensor", "enque", "deque", "freetensor"),
    "matmul_api": ("matmul", "matmultype", "settensora", "settensorb", "iterateall"),
    "datacopy_api": ("datacopy", "setglobalbuffer"),
    "vector_compute_api": ("add", "mul", "max", "min", "dup", "subs"),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", (text or "").lower())
    return [tok for tok in cleaned.split() if len(tok) >= 2 and tok not in _TOKEN_NOISE_WORDS]


def _unique_strings(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = (value or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return tuple(ordered)


def _extract_api_symbols(text: str) -> Tuple[str, ...]:
    symbols = []
    for match in _API_SYMBOL_RE.finditer(text or ""):
        symbol = match.group(0).strip(":")
        if len(symbol) >= 3 and symbol not in _API_SYMBOL_STOPWORDS:
            symbols.append(symbol)
            if "::" in symbol:
                short = symbol.split("::")[-1].strip()
                if len(short) >= 3 and short not in _API_SYMBOL_STOPWORDS:
                    symbols.append(short)
    return _unique_strings(symbols)[:24]


def _infer_operator_family(text: str) -> Tuple[str, ...]:
    haystack = (text or "").lower()
    return _unique_strings(
        family for family, hints in _OPERATOR_FAMILY_HINTS.items()
        if any(hint in haystack for hint in hints)
    )


def _infer_api_patterns(text: str) -> Tuple[str, ...]:
    haystack = (text or "").lower()
    return _unique_strings(
        pattern for pattern, hints in _API_PATTERN_HINTS.items()
        if any(hint in haystack for hint in hints)
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return vector / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _overlap_count(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    right_lookup = {item.lower() for item in right}
    return sum(1 for item in left if item.lower() in right_lookup)


def _overlap_size(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    return len({item.lower() for item in left} & {item.lower() for item in right})


def _truncate_dense_text(text: str) -> str:
    value = (text or "").strip()
    if len(value) <= _DENSE_TEXT_MAX_CHARS:
        return value
    head = value[:_DENSE_TEXT_HEAD_CHARS].rstrip()
    remaining = max(0, _DENSE_TEXT_MAX_CHARS - len(head) - len("\n...\n"))
    tail_budget = min(_DENSE_TEXT_TAIL_CHARS, remaining)
    if tail_budget <= 0:
        return head
    tail = value[-tail_budget:].lstrip()
    return f"{head}\n...\n{tail}"


def _is_dense_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


def _normalize_dense_devices(devices: Optional[Sequence[str]]) -> List[str]:
    requested = [str(device).strip() for device in (devices or ("cpu", "npu:0")) if str(device).strip()]
    if not requested:
        requested = ["cpu"]
    normalized: List[str] = []
    seen = set()
    for device in requested:
        lowered = device.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(device)
    if "cpu" not in seen:
        normalized.append("cpu")
    return normalized


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockRecord:
    index: int
    doc_id: str
    block_id: str
    path: str
    source_root: str
    block_kind: str
    context_type: str
    symbol_name: str
    operator_family: Tuple[str, ...]
    api_pattern: Tuple[str, ...]
    api_symbols: Tuple[str, ...]
    block_summary: str
    keywords: Tuple[str, ...]
    curation_tier: str
    retrieval_enabled: bool
    risk_tags: Tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    sibling_blocks: Tuple[str, ...]
    tokens: Tuple[str, ...]


@dataclass(frozen=True)
class QueryIntent:
    tokens: Tuple[str, ...]
    context_type: Optional[str]
    operator_family: Optional[str]
    api_patterns: Tuple[str, ...]
    api_symbols: Tuple[str, ...]
    keywords: Tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    block: BlockRecord
    score: float
    metadata_score: float
    bm25_score: float
    dense_score: float
    prior_score: float
    confidence: float
    confidence_label: str
    matched_branches: Tuple[str, ...]
    explanation: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class CodeSearchSnippetRetriever:
    """Search curated local code blocks with metadata + BM25 + dense retrieval."""

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
        dense_model_name: Optional[str] = None,
        dense_devices: Optional[Sequence[str]] = None,
        enable_dense: Optional[bool] = None,
    ):
        self.top_k = top_k or agent_code_search_snippet_top_k
        self.max_chars = max_chars or agent_code_search_snippet_max_chars
        self.dense_model_name = os.path.abspath(dense_model_name or rag_embedding_model)
        self.dense_devices = _normalize_dense_devices(dense_devices)
        self.enable_dense = (
            enable_dense
            if enable_dense is not None
            else os.environ.get(_ENABLE_DENSE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        )

        # Load manifest
        if manifest_path is None:
            knowledge_root = Path(agent_code_search_snippet_knowledge_root)
            manifest_path = str(knowledge_root / "code_search_snippet_manifest.json")
        self.manifest_path = manifest_path
        self._records: Optional[List[BlockRecord]] = None

        # BM25 state
        self._record_token_counts: Optional[List[Counter]] = None
        self._record_doc_lengths: Optional[List[int]] = None
        self._doc_freq: Optional[Dict[str, int]] = None
        self._avg_doc_length: float = 0.0

        # Dense state
        self._dense_model = None
        self._dense_device: Optional[str] = None
        self._dense_embeddings: Dict[int, np.ndarray] = {}
        self._dense_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading / indexing
    # ------------------------------------------------------------------
    def _load_manifest(self) -> List[BlockRecord]:
        if not os.path.exists(self.manifest_path):
            return []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            raw_blocks = json.load(f)

        records: List[BlockRecord] = []
        for i, b in enumerate(raw_blocks):
            text = (b.get("text") or "").strip()
            if not text or len(text) < _MIN_SNIPPET_CHARS:
                continue
            tokens = _unique_strings(_tokenize(f"{b.get('block_summary', '')} {text}"))
            records.append(BlockRecord(
                index=i,
                doc_id=b.get("doc_id", ""),
                block_id=b.get("block_id", ""),
                path=b.get("path", ""),
                source_root=b.get("source_root", "asc_devkit_examples"),
                block_kind=b.get("block_kind", ""),
                context_type=b.get("context_type", ""),
                symbol_name=b.get("symbol_name", ""),
                operator_family=tuple(b.get("operator_family") or []),
                api_pattern=tuple(b.get("api_pattern") or []),
                api_symbols=tuple(b.get("api_symbols") or []),
                block_summary=b.get("block_summary", ""),
                keywords=tuple(b.get("keywords") or []),
                curation_tier=b.get("curation_tier", "default"),
                retrieval_enabled=bool(b.get("retrieval_enabled", True)),
                risk_tags=tuple(b.get("risk_tags") or []),
                text=text,
                start_line=b.get("start_line", 1),
                end_line=b.get("end_line", 1),
                sibling_blocks=tuple(b.get("sibling_blocks") or []),
                tokens=tokens,
            ))
        return records

    def _get_records(self) -> List[BlockRecord]:
        if self._records is None:
            self._records = self._load_manifest()
        return self._records

    def is_available(self) -> bool:
        return bool(self._get_records())

    def available_sources(self) -> Dict[str, str]:
        return {
            "manifest_path": self.manifest_path,
            "asc_devkit": "enabled" if self.is_available() else "missing",
        }

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------
    def build_query(
        self, op_name: str, category: str, extra_context: Optional[str] = None
    ) -> str:
        parts: List[str] = []
        if extra_context:
            parts.append(extra_context)
        if op_name:
            parts.append(op_name.replace("_", " "))
        if category:
            parts.append(category)
        parts.append("Ascend C")
        return " | ".join(part for part in parts if part).strip()

    def _build_query_intent(self, query: str) -> QueryIntent:
        tokens = tuple(_tokenize(query))
        # Try to infer context_type from query
        context_type: Optional[str] = None
        qlower = (query or "").lower()
        for ct in _CONTEXT_TYPE_ORDER:
            if ct.replace("_", " ") in qlower or ct in qlower:
                context_type = ct
                break
        # operator_family
        op_family: Optional[str] = None
        for family, hints in _OPERATOR_FAMILY_HINTS.items():
            if any(hint in qlower for hint in hints):
                op_family = family
                break
        # api_patterns
        api_patterns = _infer_api_patterns(query)
        api_symbols = _extract_api_symbols(query)
        keywords = _unique_strings(_tokenize(query))
        return QueryIntent(
            tokens=tokens,
            context_type=context_type,
            operator_family=op_family,
            api_patterns=api_patterns,
            api_symbols=api_symbols,
            keywords=keywords,
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _hard_filter_indices(
        self,
        records: Sequence[BlockRecord],
        context_type: Optional[str],
        allowed_tiers: Optional[Sequence[str]] = None,
    ) -> List[int]:
        if allowed_tiers is None:
            allowed_tiers = ("preferred", "default")
        allowed_tier_set = {t.lower() for t in allowed_tiers}
        indices = []
        for i, r in enumerate(records):
            if not r.retrieval_enabled:
                continue
            if r.curation_tier.lower() not in allowed_tier_set:
                continue
            if context_type is not None and r.context_type != context_type:
                continue
            indices.append(i)
        return indices

    def _candidate_prefilter_scores(
        self,
        records: Sequence[BlockRecord],
        base_indices: Sequence[int],
        intent: QueryIntent,
        operator_family: Optional[str] = None,
        api_patterns: Optional[Sequence[str]] = None,
    ) -> Dict[int, float]:
        if not base_indices:
            return {}

        explicit_op_family = (operator_family or "").strip().lower()
        explicit_api_patterns = {p.lower() for p in (api_patterns or ())}
        inferred_api_patterns = {p.lower() for p in intent.api_patterns}

        scores: Dict[int, float] = {}
        for idx in base_indices:
            r = records[idx]
            score = 0.0

            # context_type exact match (already hard-filtered, but boost)
            if intent.context_type and r.context_type == intent.context_type:
                score += 2.0

            # operator_family
            if explicit_op_family:
                if explicit_op_family in {f.lower() for f in r.operator_family}:
                    score += 4.0
                else:
                    # Soft penalty for mismatching family
                    score -= 1.0

            # api_pattern
            if explicit_api_patterns:
                overlap = len(explicit_api_patterns & {p.lower() for p in r.api_pattern})
                if overlap:
                    score += 3.5 * overlap
            if inferred_api_patterns:
                overlap = len(inferred_api_patterns & {p.lower() for p in r.api_pattern})
                if overlap:
                    score += 2.0 * overlap

            # api_symbols
            sym_overlap = _overlap_size(r.api_symbols, intent.api_symbols)
            if sym_overlap:
                score += 2.5 * sym_overlap

            # keywords
            kw_overlap = _overlap_size(r.keywords, intent.keywords)
            if kw_overlap:
                score += 1.5 * kw_overlap

            # symbol_name / path terms
            if intent.tokens:
                q_tokens = {t.lower() for t in intent.tokens}
                if r.symbol_name and r.symbol_name.lower() in q_tokens:
                    score += 2.0
                path_tokens = _tokenize(r.path.replace("/", " "))
                path_overlap = len(q_tokens & {t.lower() for t in path_tokens})
                if path_overlap:
                    score += min(2.0, 0.5 * path_overlap)

            # general token overlap
            token_overlap = _overlap_size(r.tokens, intent.tokens)
            if token_overlap:
                score += min(2.5, 0.25 * token_overlap)

            if score > 0:
                scores[idx] = score
        return scores

    def _candidate_limit(self, result_k: int) -> int:
        requested = max(1, result_k) * _CANDIDATE_PREFILTER_MULTIPLIER
        return max(_CANDIDATE_PREFILTER_MIN, min(_CANDIDATE_PREFILTER_MAX, requested))

    def _select_candidates(
        self,
        records: Sequence[BlockRecord],
        base_indices: Sequence[int],
        intent: QueryIntent,
        result_k: int,
        operator_family: Optional[str] = None,
        api_patterns: Optional[Sequence[str]] = None,
    ) -> List[int]:
        prefilter_scores = self._candidate_prefilter_scores(
            records, base_indices, intent,
            operator_family=operator_family,
            api_patterns=api_patterns,
        )
        if not prefilter_scores:
            return base_indices[: self._candidate_limit(result_k)]

        limit = min(len(base_indices), self._candidate_limit(result_k))
        ranked = sorted(
            prefilter_scores.items(),
            key=lambda item: (-item[1], records[item[0]].path, records[item[0]].start_line),
        )
        strict = [idx for idx, score in ranked if score >= _CANDIDATE_PREFILTER_STRICT_SCORE]
        if len(strict) >= min(12, limit):
            return strict[:limit]

        relaxed = [idx for idx, score in ranked if score >= _CANDIDATE_PREFILTER_RELAXED_SCORE]
        selected = list(relaxed[:limit])
        if len(selected) < min(8, limit):
            for idx, _ in ranked:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= limit:
                        break
        return selected or base_indices[:limit]

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------
    def _ensure_bm25_index(self, records: Sequence[BlockRecord]) -> None:
        if self._record_token_counts is not None:
            return
        token_counts: List[Counter] = []
        doc_lengths: List[int] = []
        doc_freq: Dict[str, int] = {}
        total_length = 0
        for r in records:
            counts = Counter(r.tokens)
            token_counts.append(counts)
            doc_length = sum(counts.values())
            doc_lengths.append(doc_length)
            total_length += doc_length
            for token in counts:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._record_token_counts = token_counts
        self._record_doc_lengths = doc_lengths
        self._doc_freq = doc_freq
        self._avg_doc_length = (total_length / len(doc_lengths)) if doc_lengths else 0.0

    def _bm25_scores(
        self,
        records: Sequence[BlockRecord],
        query_tokens: Sequence[str],
        candidate_indices: Sequence[int],
    ) -> Dict[int, float]:
        if not query_tokens or not candidate_indices:
            return {}
        self._ensure_bm25_index(records)
        assert self._record_token_counts is not None
        assert self._record_doc_lengths is not None
        assert self._doc_freq is not None

        total_docs = len(records)
        avg_doc_length = self._avg_doc_length or 1.0
        scores: Dict[int, float] = {}
        for idx in candidate_indices:
            counts = self._record_token_counts[idx]
            doc_length = self._record_doc_lengths[idx] or 1
            score = 0.0
            for token in query_tokens:
                term_freq = counts.get(token, 0)
                if term_freq <= 0:
                    continue
                df = self._doc_freq.get(token, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denom = term_freq + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (doc_length / avg_doc_length))
                score += idf * ((term_freq * (_BM25_K1 + 1.0)) / denom)
            if score > 0:
                scores[idx] = score
        return scores

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------
    def _dense_text(self, record: BlockRecord) -> str:
        facets = [
            f"block_kind: {record.block_kind}",
            f"context_type: {record.context_type}",
        ]
        if record.operator_family:
            facets.append("operator_family: " + " ".join(record.operator_family))
        if record.api_pattern:
            facets.append("api_pattern: " + " ".join(record.api_pattern))
        if record.keywords:
            facets.append("keywords: " + " ".join(record.keywords[:8]))
        if record.api_symbols:
            facets.append("symbols: " + " ".join(record.api_symbols[:8]))
        return _truncate_dense_text("\n".join([record.block_summary] + facets + [record.text]))

    def _clear_dense_device_cache(self, device: Optional[str]) -> None:
        if not device:
            return
        try:
            import torch
        except Exception:
            return
        lowered = device.lower()
        try:
            if lowered.startswith("npu") and hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
            elif lowered.startswith("cuda") and hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_dense_model(self) -> None:
        if self._dense_model is None:
            return
        device = self._dense_device
        model = self._dense_model
        self._dense_model = None
        del model
        gc.collect()
        self._clear_dense_device_cache(device)

    def _candidate_dense_devices(self) -> List[str]:
        if self._dense_device and self._dense_device in self.dense_devices:
            return [self._dense_device] + [d for d in self.dense_devices if d != self._dense_device]
        return list(self.dense_devices)

    def _load_dense_model(self, device: Optional[str] = None):
        target_device = device or self._dense_device or (self.dense_devices[0] if self.dense_devices else "cpu")
        if self._dense_model is not None and self._dense_device == target_device:
            return self._dense_model
        if self._dense_model is not None and self._dense_device != target_device:
            self._release_dense_model()
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._dense_error = str(exc)
            print(f"[WARN] code_search_snippet dense retrieval disabled: {exc}")
            return None
        try:
            self._dense_model = SentenceTransformer(self.dense_model_name, device=target_device)
            self._dense_device = target_device
            self._dense_error = None
        except Exception as exc:
            print(f"[WARN] code_search_snippet dense model load failed on {target_device}: {exc}")
            self._dense_model = None
            self._clear_dense_device_cache(target_device)
        return self._dense_model

    def _encode_dense_inputs(self, texts: Sequence[str], *, stage: str) -> Optional[np.ndarray]:
        if not texts:
            return None
        last_error: Optional[Exception] = None
        devices = self._candidate_dense_devices()
        for index, device in enumerate(devices):
            model = self._load_dense_model(device=device)
            if model is None:
                continue
            try:
                encode_kwargs = {"show_progress_bar": False, "convert_to_numpy": True}
                if len(texts) > 1:
                    encode_kwargs["batch_size"] = min(_DENSE_BATCH_SIZE, len(texts))
                embeddings = model.encode(list(texts), **encode_kwargs)
                self._dense_error = None
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:
                last_error = exc
                self._release_dense_model()
                if index + 1 < len(devices):
                    print(f"[WARN] code_search_snippet dense {stage} failed on {device}; retrying on {devices[index + 1]}: {exc}")
                    continue
        if last_error is not None:
            self._dense_error = str(last_error)
            print(f"[WARN] code_search_snippet dense {stage} failed: {last_error}")
        return None

    def _ensure_dense_embeddings(
        self,
        records: Sequence[BlockRecord],
        candidate_indices: Sequence[int],
    ) -> Dict[int, np.ndarray]:
        if not records or not candidate_indices:
            return {}

        missing_indices = [idx for idx in candidate_indices if idx not in self._dense_embeddings]
        if missing_indices:
            embeddings = self._encode_dense_inputs(
                [self._dense_text(records[idx]) for idx in missing_indices], stage="encoding"
            )
            if embeddings is None:
                return {}
            normalized = _normalize_rows(embeddings)
            for idx, vector in zip(missing_indices, normalized):
                self._dense_embeddings[idx] = np.asarray(vector, dtype=np.float32)

        return {
            idx: self._dense_embeddings[idx]
            for idx in candidate_indices
            if idx in self._dense_embeddings
        }

    def _compute_dense_scores(
        self,
        query: str,
        records: Sequence[BlockRecord],
        candidate_indices: Sequence[int],
    ) -> Dict[int, float]:
        if not self.enable_dense or not query.strip() or not candidate_indices:
            return {}
        candidate_embeddings = self._ensure_dense_embeddings(records, candidate_indices)
        if not candidate_embeddings:
            return {}
        query_embeddings = self._encode_dense_inputs([_truncate_dense_text(query)], stage="query encode")
        if query_embeddings is None or len(query_embeddings) == 0:
            return {}
        query_vector = _normalize_vector(np.asarray(query_embeddings[0], dtype=np.float32))
        scored_indices = [idx for idx in candidate_indices if idx in candidate_embeddings]
        if not scored_indices:
            return {}
        candidate_matrix = np.stack([candidate_embeddings[idx] for idx in scored_indices], axis=0)
        similarities = candidate_matrix @ query_vector
        return {idx: float(score) for idx, score in zip(scored_indices, similarities) if score > 0}

    # ------------------------------------------------------------------
    # Scoring & ranking
    # ------------------------------------------------------------------
    def _metadata_scores(
        self,
        records: Sequence[BlockRecord],
        intent: QueryIntent,
        candidate_indices: Sequence[int],
        operator_family: Optional[str] = None,
        api_patterns: Optional[Sequence[str]] = None,
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        explicit_op_family = (operator_family or "").strip().lower()
        explicit_api_patterns = {p.lower() for p in (api_patterns or ())}
        inferred_api_patterns = {p.lower() for p in intent.api_patterns}

        for idx in candidate_indices:
            r = records[idx]
            score = 0.0

            # operator family
            if explicit_op_family:
                if explicit_op_family in {f.lower() for f in r.operator_family}:
                    score += 4.0
                else:
                    score -= 0.5

            # api patterns
            if explicit_api_patterns:
                overlap = len(explicit_api_patterns & {p.lower() for p in r.api_pattern})
                score += 3.0 * overlap
            if inferred_api_patterns:
                overlap = len(inferred_api_patterns & {p.lower() for p in r.api_pattern})
                score += 1.5 * overlap

            # api symbols
            sym_overlap = _overlap_count(r.api_symbols, intent.api_symbols)
            if sym_overlap:
                score += 2.5 * sym_overlap

            # keywords
            kw_overlap = _overlap_count(r.keywords, intent.keywords)
            if kw_overlap:
                score += 1.5 * kw_overlap

            # context_type boost
            if intent.context_type and r.context_type == intent.context_type:
                score += 2.0

            # symbol / path overlap
            if intent.tokens:
                q_tokens = {t.lower() for t in intent.tokens}
                if r.symbol_name and r.symbol_name.lower() in q_tokens:
                    score += 1.5
                path_tokens = _tokenize(r.path.replace("/", " "))
                path_overlap = len(q_tokens & {t.lower() for t in path_tokens})
                if path_overlap:
                    score += min(1.5, 0.3 * path_overlap)

            if score > 0:
                scores[idx] = score
        return scores

    def _prior_scores(
        self,
        records: Sequence[BlockRecord],
        candidate_indices: Sequence[int],
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for idx in candidate_indices:
            tier = records[idx].curation_tier
            scores[idx] = _CURATION_TIER_PRIORITY.get(tier, 1.0)
        return scores

    def _build_explanations(
        self,
        record: BlockRecord,
        intent: QueryIntent,
        operator_family: Optional[str] = None,
        api_patterns: Optional[Sequence[str]] = None,
    ) -> Tuple[str, ...]:
        explanations = []
        if intent.context_type and record.context_type == intent.context_type:
            explanations.append(f"context_type matched: {record.context_type}")
        if operator_family and operator_family.lower() in {f.lower() for f in record.operator_family}:
            explanations.append(f"operator_family matched: {operator_family}")
        explicit_api = {p.lower() for p in (api_patterns or ())}
        api_overlap = _unique_strings(
            p for p in record.api_pattern if p.lower() in explicit_api
        )
        if api_overlap:
            explanations.append("api_pattern matched: " + ", ".join(api_overlap[:4]))
        sym_overlap = _unique_strings(
            s for s in record.api_symbols if s.lower() in {a.lower() for a in intent.api_symbols}
        )
        if sym_overlap:
            explanations.append("api_symbols matched: " + ", ".join(sym_overlap[:6]))
        kw_overlap = _unique_strings(
            k for k in record.keywords if k.lower() in {t.lower() for t in intent.tokens}
        )
        if kw_overlap:
            explanations.append("keywords matched: " + ", ".join(kw_overlap[:6]))
        if record.symbol_name and record.symbol_name.lower() in {t.lower() for t in intent.tokens}:
            explanations.append(f"symbol_name matched: {record.symbol_name}")
        return tuple(explanations)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        context_type: Optional[str] = None,
        operator_family: Optional[str] = None,
        api_patterns: Optional[Sequence[str]] = None,
        source: str = "asc_devkit",
        allowed_tiers: Optional[Sequence[str]] = None,
    ) -> List[SearchResult]:
        records = self._get_records()
        if not records:
            return []

        k = top_k or self.top_k
        intent = self._build_query_intent(query)

        # If caller explicitly provides context_type, use it; otherwise fall back to intent
        effective_context_type = context_type or intent.context_type

        # Phase 1: hard filter
        base_indices = self._hard_filter_indices(records, effective_context_type, allowed_tiers)
        if not base_indices:
            # Fallback: if strict context_type yields nothing, DO NOT relax context_type per Plan.
            # But allow hidden/fallback_only tiers if explicitly requested
            if allowed_tiers is None:
                base_indices = self._hard_filter_indices(records, effective_context_type, allowed_tiers=("preferred", "default", "fallback_only", "hidden"))
            if not base_indices:
                return []

        # Phase 2: candidate selection
        candidate_indices = self._select_candidates(
            records, base_indices, intent, result_k=k,
            operator_family=operator_family or intent.operator_family,
            api_patterns=api_patterns or intent.api_patterns,
        )
        if not candidate_indices:
            return []

        # Phase 3: multi-branch scoring
        metadata_scores = self._metadata_scores(
            records, intent, candidate_indices,
            operator_family=operator_family or intent.operator_family,
            api_patterns=api_patterns or intent.api_patterns,
        )
        bm25_scores = self._bm25_scores(records, intent.tokens, candidate_indices)
        dense_scores = self._compute_dense_scores(query, records, candidate_indices)
        prior_scores = self._prior_scores(records, candidate_indices)

        # Normalize each branch to [0, 1] using softmax-ish or rank-based scaling
        def _normalize_branch(scores: Dict[int, float]) -> Dict[int, float]:
            if not scores:
                return {}
            max_score = max(scores.values())
            if max_score <= 0:
                return {idx: 0.0 for idx in scores}
            return {idx: min(1.0, score / max_score) for idx, score in scores.items()}

        norm_metadata = _normalize_branch(metadata_scores)
        norm_bm25 = _normalize_branch(bm25_scores)
        norm_dense = _normalize_branch(dense_scores)
        norm_prior = _normalize_branch(prior_scores)

        # Combine
        combined: Dict[int, float] = {}
        for idx in candidate_indices:
            score = (
                _WEIGHT_METADATA * norm_metadata.get(idx, 0.0)
                + _WEIGHT_BM25 * norm_bm25.get(idx, 0.0)
                + _WEIGHT_DENSE * norm_dense.get(idx, 0.0)
                + _WEIGHT_PRIOR * norm_prior.get(idx, 0.0)
            )
            if score > 0:
                combined[idx] = score

        if not combined:
            return []

        ranked = sorted(
            combined.items(),
            key=lambda item: (-item[1], records[item[0]].path, records[item[0]].start_line),
        )

        available_branches = sum(1 for b in (metadata_scores, bm25_scores, dense_scores) if b)
        results: List[SearchResult] = []
        for position, (idx, score) in enumerate(ranked[:k]):
            matched_branches = tuple(
                name for name, branch in (
                    ("metadata", metadata_scores),
                    ("bm25", bm25_scores),
                    ("dense", dense_scores),
                )
                if branch.get(idx, 0.0) > 0
            )
            next_score = ranked[position + 1][1] if position + 1 < len(ranked) else 0.0
            margin = score - next_score
            # Confidence heuristic
            confidence_components = [
                min(1.0, score / 0.5),
                min(1.0, len(matched_branches) / max(1, available_branches)),
                min(1.0, max(0.0, margin) / 0.1),
            ]
            confidence = sum(confidence_components) / len(confidence_components)

            results.append(SearchResult(
                block=records[idx],
                score=score,
                metadata_score=norm_metadata.get(idx, 0.0),
                bm25_score=norm_bm25.get(idx, 0.0),
                dense_score=norm_dense.get(idx, 0.0),
                prior_score=norm_prior.get(idx, 0.0),
                confidence=confidence,
                confidence_label=_confidence_label(confidence),
                matched_branches=matched_branches,
                explanation=self._build_explanations(
                    records[idx], intent,
                    operator_family=operator_family or intent.operator_family,
                    api_patterns=api_patterns or intent.api_patterns,
                ),
            ))
        return results

    # ------------------------------------------------------------------
    # Legacy-compatible retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source: str = "all",
        artifact_types: Optional[Sequence[str]] = None,
        operator_families: Optional[Sequence[str]] = None,
        source_groups: Optional[Sequence[str]] = None,
        context_type: Optional[str] = None,
        api_patterns: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Return formatted text for top-k blocks."""
        # Map legacy parameters where possible
        op_family = None
        if operator_families:
            op_family = operator_families[0]
        if context_type is None and artifact_types:
            # Heuristic mapping from old artifact_types to new context_type
            at = {a.lower() for a in artifact_types}
            if "host" in at or "tiling" in at:
                context_type = "host_tiling_src"
            elif "kernel" in at:
                context_type = "kernel_src"
            elif "opdef" in at:
                context_type = "host_op_registration"
            elif "pybind" in at:
                context_type = "python_bind_src"

        results = self.search(
            query=query,
            top_k=top_k or self.top_k,
            context_type=context_type,
            operator_family=op_family,
            api_patterns=api_patterns,
            source=source,
        )
        if not results:
            sources = self.available_sources()
            return [
                "[code_search_snippet] No matching block found. "
                f"manifest={sources['manifest_path']}, status={sources['asc_devkit']}"
            ]

        lines: List[str] = []
        lines.append(f"### Top {len(results)} Code Blocks")
        lines.append("")
        for i, r in enumerate(results, 1):
            b = r.block
            lines.append(f"--- Block {i} ---")
            lines.append(f"- score: {r.score:.3f} (metadata={r.metadata_score:.2f} bm25={r.bm25_score:.2f} dense={r.dense_score:.2f} prior={r.prior_score:.2f})")
            lines.append(f"- confidence: {r.confidence:.2f} ({r.confidence_label})")
            lines.append(f"- matched_branches: {', '.join(r.matched_branches) if r.matched_branches else 'none'}")
            lines.append(f"- block_id: {b.block_id}")
            lines.append(f"- path: {b.path}")
            lines.append(f"- block_kind: {b.block_kind}")
            lines.append(f"- context_type: {b.context_type}")
            lines.append(f"- symbol_name: {b.symbol_name}")
            lines.append(f"- operator_family: {', '.join(b.operator_family) if b.operator_family else 'none'}")
            lines.append(f"- api_pattern: {', '.join(b.api_pattern) if b.api_pattern else 'none'}")
            lines.append(f"- curation_tier: {b.curation_tier}")
            lines.append(f"- risk_tags: {', '.join(b.risk_tags) if b.risk_tags else 'none'}")
            for expl in r.explanation:
                lines.append(f"- why_matched: {expl}")
            if b.sibling_blocks:
                lines.append(f"- sibling_blocks: {', '.join(b.sibling_blocks[:3])}")
            lines.append("")
            lines.append(f"```cpp")
            # Truncate if needed
            code = b.text
            if len(code) > self.max_chars:
                code = code[: self.max_chars] + "\n// ... truncated"
            lines.append(code)
            lines.append("```")
            lines.append("")
        return ["\n".join(lines)]


# Backward-compatible alias for external imports
CodeSearchSnippetResult = SearchResult
