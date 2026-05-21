"""Hand-authored NL patches for canonical memories (scenario scope / truncation fixes)."""

from __future__ import annotations

from typing import Dict

# memory_id -> replacement natural_language (English, single sentence, When...do not...instead...)
NL_PATCHES: Dict[str, str] = {
    "0342a1a9-992b-4d64-a4b3-9426f5c27408": (
        "When NPU eval reports vector core timeout (507034) after masked_cumsum compiles, "
        "and host TilingFunc calls GetInputShape, do not treat GetInputShape as returning const gert::Shape*; "
        "instead use const gert::StorageShape* and GetOriginShape() in TilingFunc (gert::TilingContext) only."
    ),
    "7340946f-9a7f-4450-89ca-f627a85315be": (
        "When opbuild fails in InferShape with cannot convert const gert::Shape* to const gert::StorageShape* "
        "on GetInputShape, do not use StorageShape* or SetOriginShape; "
        "instead use const gert::Shape* with SetDimNum and SetDim in InferShape (gert::InferShapeContext)."
    ),
    "cd8066f0-7e8c-4350-b439-2d77be28d0f8": (
        "When opbuild fails in TilingFunc at op_host/argmax_over_a_dimension_custom.cpp with cannot convert "
        "StorageShape* to Shape* on GetInputShape, do not declare the result as const gert::Shape*; "
        "instead use const gert::StorageShape* and read dimensions via GetOriginShape()."
    ),
    "ddf7fbd3-8a3b-46ad-a77c-260ac91db646": (
        "When opbuild fails in masked_cumsum host TilingFunc with cannot convert StorageShape* to Shape* "
        "and missing GetOriginShape on Shape, do not assign GetInputShape to const gert::Shape*; "
        "instead use const gert::StorageShape*, read dims via GetOriginShape(), and use the correct attr API for tiling fields."
    ),
    "d7872c76-18c8-424b-afdd-62bc2b9bb054": (
        "When the operator txt bundle raises ValueError for missing host_operator_src, kernel_src, "
        "and python_bind_src blocks, do not emit a txt missing any of those sections; "
        "instead include all three blocks with valid code in the generated operator txt."
    ),
    "5631a2fa-b6a6-4104-9617-c7cb2139acd0": (
        "When CPack INSTALL cannot find op_kernel/binary/config during vanilla_rnn_hidden packaging, "
        "do not keep an unused tiling field K; instead remove K from tiling data and size the w_i2h buffer using I+H only."
    ),
    "ca69bb7f-ac7a-4b8c-9d74-49bb85207029": (
        "When NPU eval reports output shape mismatch for max_pooling_3d (expected vs actual D/H/W), "
        "do not assume tiling fields D_out/H_out/W_out are already correct; "
        "instead recompute them in TilingFunc as (dim + 2*padding - kernel)/stride + 1 for each spatial axis."
    ),
}
