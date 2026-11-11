# SPDX-License-Identifier: Apache-2.0
"""Bit-exact gate for the lane-native int8 MFMA primitives (gfx908)."""

from aiter.ops.int8_native import selftest

if __name__ == "__main__":
    print(selftest())
