from typing import Any, Dict, List


DTYPE_BYTES: Dict[str, int] = {
    "float": 4,
    "float32": 4,
    "half": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int32": 4,
    "int16": 2,
    "short": 2,
    "int8": 1,
    "uint8": 1,
    "char": 1,
    "double": 8,
    "float64": 8,
}

ALIGNMENT_BYTES = 256
MAX_REPEAT_TIMES = 255
DEFAULT_UB_CAPACITY = 196608

GENERIC_TILING_CLASSES = {"elementwise", "broadcast"}
ALL_OPERATOR_CLASSES = GENERIC_TILING_CLASSES | {
    "reduction",
    "conversion",
    "random",
    "matmul",
    "convolution",
    "nn",
    "unknown",
}

CATEGORY_CLASS_MAP = {
    "activation": "elementwise",
    "elementwise": "elementwise",
    "broadcast": "broadcast",
    "reduction": "reduction",
    "reduce": "reduction",
    "conversion": "conversion",
    "transform": "conversion",
    "transpose": "conversion",
    "concat": "conversion",
    "split": "conversion",
    "random": "random",
    "matmul": "matmul",
    "matrix": "matmul",
    "convolution": "convolution",
    "conv": "convolution",
    "nn": "nn",
    "normalization": "nn",
    "pooling": "nn",
}

CLASS_KEYWORDS: Dict[str, List[str]] = {
    "nn": [
        "layer_norm",
        "layernorm",
        "group_norm",
        "groupnorm",
        "batch_norm",
        "batchnorm",
        "softmax",
        "log_softmax",
        "rmsnorm",
        "flashattention",
        "flash_attention",
        "max_pool",
        "avg_pool",
        "pooling",
        "maxpool",
        "avgpool",
    ],
    "matmul": ["matmul", "bmm", "gemm"],
    "convolution": ["conv", "conv2d", "convolution", "depthwise_conv", "depthwiseconv"],
    "conversion": ["transpose", "permute", "concat", "split", "reshape", "layout"],
    "random": ["random", "dropout", "uniform", "bernoulli", "randn"],
    "reduction": [
        "argmax",
        "argmin",
        "topk",
        "reduction",
        "reduce",
        "mean",
        "sum",
        "amax",
        "amin",
    ],
    "broadcast": ["broadcast", "bias", "expand"],
    "elementwise": [
        "elementwise",
        "relu",
        "gelu",
        "elu",
        "softplus",
        "swish",
        "hardsigmoid",
        "hardtanh",
        "leaky_relu",
        "selu",
        "add",
        "sub",
        "mul",
        "div",
    ],
}

BLACKLISTED_OPERATORS: Dict[str, Dict[str, Any]] = {
    "layer_norm": {
        "operator_class": "nn",
        "reason": "layer_norm requires operator-specific multi-pass reduction tiling",
        "required_inputs": [
            "operator-specific reduction strategy",
            "buffer plan",
            "multi-pass design",
        ],
    },
    "group_norm": {
        "operator_class": "nn",
        "reason": "group_norm requires operator-specific normalization tiling",
        "required_inputs": [
            "group-wise reduction strategy",
            "buffer plan",
            "multi-pass design",
        ],
    },
    "batch_norm": {
        "operator_class": "nn",
        "reason": "batch_norm requires operator-specific normalization tiling",
        "required_inputs": [
            "statistic accumulation strategy",
            "buffer plan",
            "multi-pass design",
        ],
    },
    "softmax": {
        "operator_class": "nn",
        "reason": "softmax requires reduction-aware normalization tiling",
        "required_inputs": [
            "reduction strategy",
            "buffer plan",
            "exp/sum multi-pass design",
        ],
    },
    "log_softmax": {
        "operator_class": "nn",
        "reason": "log_softmax requires reduction-aware normalization tiling",
        "required_inputs": [
            "reduction strategy",
            "buffer plan",
            "exp/log multi-pass design",
        ],
    },
    "argmax": {
        "operator_class": "reduction",
        "reason": "argmax requires operator-specific reduction tiling",
        "required_inputs": ["reduction strategy", "index tracking", "buffer plan"],
    },
    "argmin": {
        "operator_class": "reduction",
        "reason": "argmin requires operator-specific reduction tiling",
        "required_inputs": ["reduction strategy", "index tracking", "buffer plan"],
    },
    "topk": {
        "operator_class": "reduction",
        "reason": "topk requires operator-specific selection tiling",
        "required_inputs": ["selection strategy", "workspace plan", "buffer plan"],
    },
    "max_pool": {
        "operator_class": "nn",
        "reason": "max_pool requires window-aware pooling tiling",
        "required_inputs": ["window traversal strategy", "padding policy", "buffer plan"],
    },
    "avg_pool": {
        "operator_class": "nn",
        "reason": "avg_pool requires window-aware pooling tiling",
        "required_inputs": ["window traversal strategy", "padding policy", "buffer plan"],
    },
    "matmul": {
        "operator_class": "matmul",
        "reason": "matmul requires matrix-specific block tiling",
        "required_inputs": ["matrix blocking strategy", "L0/L1 buffer plan", "load scheduling"],
    },
    "bmm": {
        "operator_class": "matmul",
        "reason": "bmm requires matrix-specific block tiling",
        "required_inputs": ["matrix blocking strategy", "L0/L1 buffer plan", "load scheduling"],
    },
    "gemm": {
        "operator_class": "matmul",
        "reason": "gemm requires matrix-specific block tiling",
        "required_inputs": ["matrix blocking strategy", "L0/L1 buffer plan", "load scheduling"],
    },
}