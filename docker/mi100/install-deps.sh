#!/usr/bin/env bash
# Run from the checkout with a fresh Python 3.12 .venv.
set -euo pipefail
py=.venv/bin/python
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
index=https://download.pytorch.org/whl
cat > "$work/constraints.txt" <<'PINS'
torch==2.11.0+rocm7.1
torchvision==0.26.0+rocm7.1
torchaudio==2.11.0+rocm7.1
triton-rocm==3.6.0
PINS
install() {
  "$py" -m pip install --dry-run "$@" && "$py" -m pip install "$@"
}
install -c "$work/constraints.txt" --extra-index-url "$index/rocm7.1" torch torchvision torchaudio
install -c "$work/constraints.txt" -r requirements/rocm.txt 'cmake>=3.26,<4' ninja pybind11 einops wheel /opt/rocm/share/amd_smi
# Some optional packages depend on the PyPI triton distribution. Both
# distributions install the same import tree; restore the ROCm owner last.
if "$py" -m pip show triton >/dev/null 2>&1; then
  "$py" -m pip uninstall -y triton
  install --force-reinstall --no-deps --index-url "$index/rocm7.1" triton-rocm==3.6.0
fi
"$py" -m pip download --no-deps --index-url "$index/rocm7.2" --dest "$work" torch==2.13.0+rocm7.2
"$py" - "$work" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import torch

lib = Path(torch.__file__).parent / "lib"
expected = {
    "libamdhip64.so": "220bc845a73e278c42e63e8dba61890ea2fdf3959941921b9447631f5675ced7",
    "libhsa-runtime64.so": "843e549d187929cd884bad148a2faee67159eef1da67adf6320af7c37d9d7739",
}
with ZipFile(next(Path(sys.argv[1]).glob("torch-*.whl"))) as wheel:
    for name, digest in expected.items():
        data = wheel.read(f"torch/lib/{name}")
        if hashlib.sha256(data).hexdigest() != digest:
            sys.exit(f"Unexpected ROCm runtime in donor wheel: {name}")
        dest = lib / name
        backup = lib / (name + ".wheel-rocm7.1")
        if not backup.exists():
            shutil.copy2(dest, backup)
        staged = lib / (name + ".new")
        staged.write_bytes(data)
        staged.chmod(0o755)
        staged.replace(dest)
PY
"$py" -c 'import torch, triton; assert torch.__version__ == "2.11.0+rocm7.1" and triton.__version__ == "3.6.0"'
