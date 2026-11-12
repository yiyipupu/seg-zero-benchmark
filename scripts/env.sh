# Source this file from the repository root in each new terminal.
export SEG_ROOT=${SEG_ROOT:-/data/yukun/Seg}
export CONDA_PKGS_DIRS=$SEG_ROOT/cache/conda_pkgs
export PIP_CACHE_DIR=$SEG_ROOT/cache/pip
export HF_HOME=$SEG_ROOT/cache/huggingface
export TORCH_HOME=$SEG_ROOT/cache/torch
export TMPDIR=$SEG_ROOT/cache/tmp
export MPLBACKEND=Agg
mkdir -p "$SEG_ROOT"/{env,cache,checkpoints,datasets,outputs,weights} "$TMPDIR"
