#!/usr/bin/env bash
# Run from the checkout with a fresh Python 3.12 .venv.
set -euo pipefail
py=.venv/bin/python
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
case "${1:-}" in
  --rocm10.2)
    index=https://stable.repo.amd.com/rocm/whl-next/
    cat > "$work/constraints.txt" <<'PINS'
torch==2.15.0+rocm10.2.0
torchvision==0.29.2+rocm10.2.0
torchaudio==2.11.1+rocm10.2.0
triton==3.9.0+git0a6e13e6.rocm10.2.0
PINS
    ;;
  "")
    index=https://download.pytorch.org/whl
    cat > "$work/constraints.txt" <<'PINS'
torch==2.11.0+rocm7.1
torchvision==0.26.0+rocm7.1
torchaudio==2.11.0+rocm7.1
triton-rocm==3.6.0
PINS
    ;;
  *) echo "Usage: $0 [--rocm10.2]" >&2; exit 2 ;;
esac
install() {
  "$py" -m pip install --dry-run "$@" && "$py" -m pip install "$@"
}
if [[ "${1:-}" == --rocm10.2 ]]; then
  install -c "$work/constraints.txt" --index-url "$index" --extra-index-url https://pypi.org/simple \
    'torch[device-gfx908]' 'torchvision[device-gfx908]' torchaudio triton
else
  install -c "$work/constraints.txt" --extra-index-url "$index/rocm7.1" torch torchvision torchaudio
fi
install -c "$work/constraints.txt" -r requirements/rocm.txt 'cmake>=3.26,<4' ninja pybind11 einops wheel
smi_dir="${ROCM_PATH:-/opt/rocm}/share/amd_smi"
if [[ "${1:-}" == --rocm10.2 ]]; then
  # Torch loads the SDK wheel's libamd_smi. A second copy from the system
  # bindings can report zero processors even when HIP sees both MI100s.
  smi_dir=$("$py" -c 'from pathlib import Path; import _rocm_sdk_core; print(Path(next(iter(_rocm_sdk_core.__path__))) / "share/amd_smi")')
fi
if [[ -f "$smi_dir/pyproject.toml" || -f "$smi_dir/setup.py" ]]; then
  install -c "$work/constraints.txt" "$smi_dir"
else
  # ROCm 10.2 ships the importable package without Python build metadata.
  test -f "$smi_dir/amdsmi/__init__.py"
  site_dir=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  printf '%s\n' "$smi_dir" > "$site_dir/amd_smi.pth"
fi
"$py" -c 'import amdsmi'
if [[ "${1:-}" == --rocm10.2 ]]; then
  "$py" -c 'import torch, torchvision, torchaudio, triton; assert torch.__version__ == "2.15.0+rocm10.2.0"'
  exit 0
fi
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
