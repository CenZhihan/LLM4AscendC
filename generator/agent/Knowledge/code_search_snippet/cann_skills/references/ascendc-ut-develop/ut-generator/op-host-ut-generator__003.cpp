// BroadcastCompileInfo（广播类算子）
#include "atvoss/broadcast/broadcast_tiling.h"
Ops::Base::BroadcastCompileInfo compileInfo = {64, 245760};

// 自定义 CompileInfo（如 AddCompileInfo）
#include "../../../../op_host/arch35/add_tiling_arch35.h"
optiling::AddCompileInfo compileInfo = {64, 245760};
