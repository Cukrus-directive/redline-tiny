# Activate a compile-only IRON env on a Linux build box (example: mlir-aie checked out in /opt/mlir-aie-<tree>,
# xclbinutil in /opt/npu/bin).
# usage: source build/env.sh [tree]      tree defaults to v1.4.4, the version the shipped programs were built with
_t=${1:-v1.4.4}
cd /opt/mlir-aie-$_t && source ./iron_env.sh >/dev/null && cd - >/dev/null
export PATH=/opt/npu/bin:$PATH
export AIE_XCLBINUTIL=/opt/npu/bin/xclbinutil
export MLIR_AIE_SRC=/opt/mlir-aie-$_t
export NPU_TAG="${_t//./}"
echo "IRON $_t: mlir_aie $(python3 -c 'from importlib.metadata import version as v; print(v("mlir_aie"))') | peano $PEANO_INSTALL_DIR | tag $NPU_TAG"
