#!/usr/bin/env python3
"""Check that a classic HIP D2H store's completion event follows its DMA.

The production executor uses hipMemcpyAsync(..., stream=0). A side-stream
completion event is insufficient on gfx908: the CPU tier can publish a page
and release its GPU source while the default-stream copy is still in flight.
"""

import ctypes
import time

import torch


def main() -> None:
    torch.cuda.set_device(0)
    size = 128 * 1024 * 1024
    src = torch.full((size,), 0x5A, dtype=torch.uint8, device="cuda")
    dst = torch.empty((size,), dtype=torch.uint8, pin_memory=True)
    dst.zero_()
    torch.cuda.synchronize()

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_int,
                                   ctypes.c_void_p]
    hip.hipMemcpyAsync.restype = ctypes.c_int
    side = torch.cuda.Stream()
    default = torch.cuda.default_stream()
    assert default.cuda_stream == 0, (
        "PyTorch's default stream must match HIP stream 0"
    )

    def copy() -> None:
        rc = hip.hipMemcpyAsync(
            ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()),
            ctypes.c_size_t(size), 2, ctypes.c_void_p(0),
        )
        if rc:
            raise RuntimeError(f"hipMemcpyAsync failed: {rc}")

    # Current production ordering before the fix: the end event sits on the
    # side stream even though the DMA runs on stream 0.
    begin = torch.cuda.Event(enable_timing=True)
    old_end = torch.cuda.Event(enable_timing=True)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        begin.record(side)
        copy()
        old_end.record(side)
    t0 = time.perf_counter()
    old_end.synchronize()
    old_event_s = time.perf_counter() - t0
    copy_pending_at_old_end = not default.query()
    t1 = time.perf_counter()
    default.synchronize()
    remaining_dma_s = time.perf_counter() - t1

    # Correct ordering: default stream waits for the transfer stream's source
    # readiness and owns the end event. Publication follows real DMA completion.
    dst.zero_()
    begin2 = torch.cuda.Event(enable_timing=True)
    new_end = torch.cuda.Event(enable_timing=True)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        begin2.record(side)
        default.wait_event(begin2)
        copy()
        new_end.record(default)
    t2 = time.perf_counter()
    new_end.synchronize()
    new_event_s = time.perf_counter() - t2
    copy_pending_at_new_end = not default.query()
    good_data = dst[0].item() == 0x5A and dst[-1].item() == 0x5A

    print(f"old event wait={old_event_s * 1e3:.3f}ms "
          f"DMA pending={copy_pending_at_old_end} "
          f"remaining DMA={remaining_dma_s * 1e3:.3f}ms")
    print(f"new event wait={new_event_s * 1e3:.3f}ms "
          f"DMA pending={copy_pending_at_new_end} data_ok={good_data}")
    assert not copy_pending_at_new_end and good_data


if __name__ == "__main__":
    main()
