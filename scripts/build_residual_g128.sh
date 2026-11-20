#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-vllm/_residual_g128.so}"
mkdir -p "$(dirname "$output")"
tmp=$(mktemp "$(dirname "$output")/.residual_g128.XXXXXXXX.so")
trap 'rm -f -- "$tmp"' EXIT
/opt/rocm/bin/hipcc --offload-arch=gfx908 -O3 -ffp-contract=off -shared -fPIC \
  csrc/custom_all_reduce_g128.hip -o "$tmp"
chmod 755 "$tmp"
mv -f -- "$tmp" "$output"
