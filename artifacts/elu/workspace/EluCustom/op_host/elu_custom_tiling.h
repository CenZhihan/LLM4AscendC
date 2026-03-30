
#include "register/tilingdata_base.h"

namespace optiling {
// 注册TilingData结构，包含长度、分块数以及alpha参数
BEGIN_TILING_DATA_DEF(EluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, alpha);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EluCustom, EluCustomTilingData)
}
