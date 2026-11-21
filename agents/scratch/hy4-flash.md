# HY4 Flash reproducibility notes

The immutable upstream model backport is vLLM commit `b067b6f6fa509a8c98c5a79d75b2aa3742d539c6`. The conversion reference used Transformers commit `b865850d05326cf20e306b31f14b0027de802da3`. The original Tencent checkpoint was `b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f`.

With hipFile 0.3.0, the original model volume returned ENODEV in strict mode on all four cards for aligned 4 KiB reads and writes. Its successful compatibility-mode probe registered 32,768 read bytes and 16,384 write bytes in the fallback counter. A native NVMe read registered 16,384 fast bytes and zero fallback. Use the physical-transfer counters to identify a suitable conversion workspace after changing the disk layout or driver.

The native run records, cold and warm Compose logs, per-rank traces and fixed teacher distributions are under the ignored `agents/tmp/hy4/` directory. The recipe and public ledger contain the release settings and measured results.
