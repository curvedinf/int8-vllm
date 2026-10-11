"""CPU-only tests: .venv/bin/python -m unittest discover -s scripts -p test_bench_agent_mixed.py"""

import asyncio
import importlib.util
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web

from bench_agent_mixed import Run, merge_delta, nginx_down, sse, summarize
from analyze_agent_mixed import merged
from analyze_agent_profile import category, phase, stats, union_ns
from kineto_to_agent_csv import convert


class Helpers(unittest.TestCase):
    def test_lifecycle_metadata_does_not_serialize_device_or_array_fields(self):
        spec = importlib.util.spec_from_file_location(
            "lifecycle_test", Path(__file__).parents[1] / "vllm/profiler/lifecycle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        obj = SimpleNamespace(num_scheduled_tokens=object())
        self.assertEqual(module._metadata("runner.prepare_attn", (None, obj)), {})
        self.assertEqual(module._metadata("worker.execute_model", (None, obj)), {})
        fn = lambda: 42
        self.assertIs(module.lifecycle("disabled")(fn), fn)

    def test_burst_rate_excludes_first_burst(self):
        r = summarize({"sent_ns": 0, "tokens": [[10**9, [1, 2, 3]],
                                               [2*10**9, [4, 5]],
                                               [4*10**9, [6]]]})
        self.assertEqual(r["decode_tok_s"], 1)
        self.assertEqual(r["ttft_s"], 1)
        self.assertEqual(r["stream_gap_s"]["max"], 2)

    def test_tool_fragments(self):
        m = {}
        merge_delta(m, {"tool_calls": [{"index": 0, "id": "abc", "function": {
            "name": "read_file", "arguments": '{"path":'}}]})
        merge_delta(m, {"tool_calls": [{"index": 0, "function": {
            "arguments": '"README.md"}'}}], "reasoning": "why"})
        self.assertEqual(json.loads(m["tool_calls"][0]["function"]["arguments"]),
                         {"path": "README.md"})
        self.assertEqual(m["reasoning"], "why")

    def test_nginx_fail_closed(self):
        for code in (0, 2):
            with patch("bench_agent_mixed.subprocess.run",
                       return_value=SimpleNamespace(returncode=code)):
                with self.assertRaises(RuntimeError):
                    nginx_down()

    def test_profile_interval_accounting(self):
        self.assertEqual(union_ns([(0, 10), (5, 15), (20, 22)]), 17)
        self.assertEqual(merged([(0, 10), (5, 15), (20, 22)]), [[0, 15], [20, 22]])
        self.assertEqual(category("custom_all_reduce_kernel"), "collective")
        self.assertEqual(phase({"requests": {"a": {"output": 1},
                                            "b": {"output": 0}}}), "mixed")
        self.assertEqual(stats([1, 2])["p50_s"], 1.5)
        self.assertEqual(stats([1, 2])["p95_s"], 2)

    def test_kineto_clock_conversion(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "test-123.json.gz"
            with gzip.open(trace, "wt") as file:
                json.dump({"baseTimeNanoseconds": 10000, "traceEvents": [{
                    "ph": "X", "cat": "kernel", "name": "test_kernel", "pid": 0,
                    "ts": 12.5, "dur": 2.5,
                    "args": {"device": 0, "correlation": 4}}]}, file)
            (root / "test-123.clock.json").write_text(json.dumps({
                "pid": 123, "start": {"wall_minus_monotonic_ns": 2000}}))
            convert(trace, root / "csv")
            with (root / "csv/123_kernel_trace.csv").open() as file:
                row = next(csv.DictReader(file))
            self.assertEqual(int(row["Start_Timestamp"]), 20500)
            self.assertEqual(int(row["End_Timestamp"]), 23000)
            convert(trace, root / "csv", append=True)
            with (root / "csv/123_kernel_trace.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)

    def test_gpu_capture_rotation_is_bounded(self):
        spec = importlib.util.spec_from_file_location(
            "lifecycle_test", Path(__file__).parents[1] / "vllm/profiler/lifecycle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._gpu_profiler = object()
        module._gpu_base_tag = "test"
        with patch.dict("os.environ", {"VLLM_LIFECYCLE_GPU_CHUNK_STEPS": "2"}), \
             patch.object(module, "_stop_gpu_chunk", return_value={"path": "first"}) as stop, \
             patch.object(module, "_start_gpu_chunk") as start, \
             patch.object(module, "_gpu_chunk_barrier") as barrier:
            module._rotate_gpu_profile("worker.execute_model")
            module._rotate_gpu_profile("worker.sample_tokens")
            stop.assert_not_called()
            module._rotate_gpu_profile("worker.sample_tokens")
            stop.assert_called_once()
            start.assert_called_once_with("test-part0001")
            self.assertEqual(module._gpu_steps, 0)
            self.assertEqual(module._gpu_chunks, [{"path": "first"}])
            barrier.assert_called_once()

    def test_export_releases_cyclic_profiler_before_next_capture(self):
        import weakref
        spec = importlib.util.spec_from_file_location(
            "lifecycle_test", Path(__file__).parents[1] / "vllm/profiler/lifecycle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Profile:
            def __init__(self):
                self.action_map = {"stop": self.stop}
            def stop(self):
                pass
            def export_chrome_trace(self, path):
                Path(path).write_text('{"traceEvents": []}')
        with tempfile.TemporaryDirectory() as directory:
            module._DIRECTORY = directory
            module._gpu_tag = "test"
            module._gpu_clock = module._clock_offset()
            profiler = Profile()
            reference = weakref.ref(profiler)
            module._gpu_profiler = profiler
            del profiler
            with patch.dict("sys.modules", {"torch": SimpleNamespace(
                    cuda=SimpleNamespace(synchronize=lambda: None))}):
                result = module._stop_gpu_chunk()
            self.assertIsNone(reference())
            self.assertFalse(list(Path(directory).glob("gpu/*.json.tmp")))
            with gzip.open(result["path"], "rt") as file:
                self.assertEqual(json.load(file), {"traceEvents": []})

    def test_lifecycle_enabled_span(self):
        spec = importlib.util.spec_from_file_location(
            "lifecycle_test", Path(__file__).parents[1] / "vllm/profiler/lifecycle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "active"
            gate.touch()
            module._DIRECTORY, module._GATE = directory, str(gate)
            @module.lifecycle("runner.prepare_attn")
            def fn(obj, batch):
                return 123
            self.assertEqual(fn(None, SimpleNamespace(num_scheduled_tokens=object())), 123)
            module._stream.close()
            rows = [json.loads(line) for f in Path(directory).glob("*.jsonl")
                    for line in f.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]["end_ns"], rows[0]["start_ns"])


class Client(unittest.IsolatedAsyncioTestCase):
    async def test_sse_multiline(self):
        async def lines():
            for line in [b": keepalive\n", b"data: a\n", b"data:b\r\n", b"\n",
                         b"data: [DONE]\n\n"]:
                yield line
        self.assertEqual([x async for x in sse(lines())], ["a\nb", "[DONE]"])

    async def test_http_lifecycle(self):
        import aiohttp
        async def handler(request):
            body = await request.json()
            if body.get("reject"):
                return web.Response(status=400, text="bad request")
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await r.prepare(request)
            await r.write(("data: " + json.dumps({"prompt_token_ids": [12345]*40000,
                                                "choices": []}) + "\n\n").encode())
            payload = "data: " + json.dumps({"choices": [{"token_ids": [42],
                "delta": {"content": "hello"}, "finish_reason": "stop"}]}) + "\n\n"
            # Deliberately fragment a UTF-8 SSE event across network writes.
            for chunk in (payload[:11], payload[11:]):
                await r.write(chunk.encode())
            if not body.get("truncate"):
                await r.write(b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n')
                await r.write(b"data: [DONE]\n\n")
            await r.write_eof()
            return r
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                run = Run(SimpleNamespace(output=str(Path(directory) / "run"),
                    max_outstanding=2, url=f"http://127.0.0.1:{port}"), {})
                try:
                    async with aiohttp.ClientSession(read_bufsize=4*1024*1024) as session:
                        good = await run.request(session, {}, "good")
                        bad = await run.request(session, {"truncate": True}, "truncated")
                        rejected = await run.request(session, {"reject": True}, "rejected")
                    self.assertEqual(good["status"], "complete")
                    self.assertEqual(good["output_tokens_observed"], 1)
                    self.assertEqual(bad["status"], "failed")
                    self.assertEqual(rejected["http_status"], 400)
                finally:
                    run.events.close()
        finally:
            await runner.cleanup()

    async def test_deadline_keeps_failures_and_removes_gate(self):
        async def stalled(self, session, spec):
            await asyncio.sleep(10)
        async def monitor(self, session):
            await asyncio.sleep(10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=str(root / "run"), max_outstanding=1,
                trace_gate=str(root / "gate"), request_timeout=1, duration=.02,
                replay_from=None, drain_seconds=0)
            run = Run(args, {"conversations": [{"id": 0}], "output_budgets": [32, 32]})
            with patch("bench_agent_mixed.nginx_down"), patch.dict(
                "os.environ", {"VLLM_API_KEY": "test"}), patch.object(
                Run, "conversation", stalled), patch.object(Run, "metrics", monitor):
                await run.run()
            self.assertFalse((root / "gate").exists())
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(len(summary), 2)
            self.assertTrue(all(r["status"].startswith("not_started") for r in summary))


if __name__ == "__main__":
    unittest.main()
