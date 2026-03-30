
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF_STRUCT(LogSoftMaxTiling, logSoftmaxTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSoftmaxCustom, LogSoftmaxCustomTilingData)
}
