#!/bin/bash
echo "[ascend910b] Generating EluCustom_c435979d8888339fb1afb9e2d55d6e76 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=elu_custom --input_param=/root/LLM4AscendC/artifacts/elu/workspace/EluCustom/build_out/op_kernel/EluCustom_ascend910b/bin_param/EluCustom_c435979d8888339fb1afb9e2d55d6e76_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/EluCustom_c435979d8888339fb1afb9e2d55d6e76.json ; then
  echo "$2/EluCustom_c435979d8888339fb1afb9e2d55d6e76.json not generated!"
  exit 1
fi

if ! test -f $2/EluCustom_c435979d8888339fb1afb9e2d55d6e76.o ; then
  echo "$2/EluCustom_c435979d8888339fb1afb9e2d55d6e76.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating EluCustom_c435979d8888339fb1afb9e2d55d6e76 Done"
