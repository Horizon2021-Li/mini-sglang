# Usage: source /home/horizon/Code/ai_infra/mini-sglang/activate-minisgl.sh
_minisgl_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_minisgl_root}/.venv/bin/activate"
export CUDA_HOME="${_minisgl_root}/.cuda-12.9"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${_minisgl_root}/.system-libs/usr/lib/x86_64-linux-gnu:${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _minisgl_root
