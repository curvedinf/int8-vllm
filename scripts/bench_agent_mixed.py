#!/usr/bin/env python3
"""Bounded, dependency-aware long-context chat benchmark (no background load).

Prepare a frozen manifest once, then replay it on isolated baseline/profile
boots. Arrival times are fixed; follow-ups depend on the preceding response.
The saved request bodies allow a later exact open-loop replay if desired.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import aiohttp


TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a project file",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}, "required": ["path"]}}}]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def nginx_down():
    """Fail closed, including manually launched nginx outside systemd."""
    result = subprocess.run(["pgrep", "-x", "nginx"], capture_output=True)
    if result.returncode != 1:
        raise RuntimeError("Nginx must be stopped before this benchmark")


def system_snapshot():
    snapshot = {"processes": {}, "gpu": {}}
    pids = subprocess.run(["pgrep", "-f", "^VLLM::"], capture_output=True,
                          text=True).stdout.split()
    for pid in pids:
        values = {}
        for field in ("status", "io", "schedstat"):
            try:
                values[field] = Path(f"/proc/{pid}/{field}").read_text()
            except (OSError, PermissionError):
                pass
        snapshot["processes"][pid] = values
    for device in Path("/sys/class/drm").glob("card[0-9]*/device"):
        values = {}
        for field in ("gpu_busy_percent", "mem_busy_percent", "pp_dpm_sclk"):
            try:
                values[field] = (device / field).read_text().strip()
            except OSError:
                pass
        for hwmon in device.glob("hwmon/hwmon*"):
            for field in ("power1_average", "power1_cap", "temp1_input"):
                try:
                    values[field] = (hwmon / field).read_text().strip()
                except OSError:
                    pass
        if values:
            snapshot["gpu"][device.parent.name] = values
    return snapshot


def configuration_snapshot():
    root = Path(__file__).resolve().parents[1]
    result = {"diff_sha256": hashlib.sha256(subprocess.check_output(
        ["git", "diff", "HEAD"], cwd=root)).hexdigest()}
    result["aiter_git"] = subprocess.check_output(
        ["git", "-C", str(root.parent / "aiter"), "rev-parse", "HEAD"],
        text=True).strip()
    try:
        pid = int((root / "logs/serve_recipe_qwen38/server.pid").read_text())
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        result["server_environment"] = {}
        for item in env:
            key, _, value = item.decode(errors="replace").partition("=")
            if key.startswith(("VLLM_", "ROCM_", "HIP_", "PYTORCH_", "GPU_")):
                if not any(word in key for word in ("KEY", "TOKEN", "PASSWORD")):
                    result["server_environment"][key] = value
        result["server_pid"] = pid
    except OSError as exc:
        result["server_snapshot_error"] = str(exc)
    sources = list((root / "scripts").glob("*agent*py")) + [
        root / "vllm/profiler/lifecycle.py", root / "scripts/serve_recipe_qwen38.sh"]
    result["test_sources"] = {str(p.relative_to(root)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return result


async def sse(content):
    """aiohttp iteration yields lines, independent of TCP fragmentation."""
    data = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            yield "\n".join(data)
            data = []
    if data:
        yield "\n".join(data)


def merge_delta(message, delta):
    for key in ("content", "reasoning", "reasoning_content"):
        if delta.get(key):
            message[key] = message.get(key, "") + delta[key]
    for part in delta.get("tool_calls") or []:
        index = part["index"]
        calls = message.setdefault("tool_calls", [])
        while len(calls) <= index:
            calls.append({"id": "", "type": "function", "function": {
                "name": "", "arguments": ""}})
        call = calls[index]
        if part.get("id"):
            call["id"] += part["id"]
        for key in ("name", "arguments"):
            call["function"][key] += (part.get("function") or {}).get(key) or ""


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def summarize(result):
    events = result["tokens"]
    gaps = [(b[0] - a[0]) / 1e9 for a, b in zip(events, events[1:])]
    result["output_tokens_observed"] = sum(len(e[1]) for e in events)
    reported = (result.get("usage") or {}).get("completion_tokens")
    result["token_accounting_delta"] = (
        result["output_tokens_observed"] - reported if reported is not None else None)
    result["ttft_s"] = ((events[0][0] - result["sent_ns"]) / 1e9
                        if events else None)
    result["client_slot_wait_s"] = (
        (result["sent_ns"] - result["ready_ns"]) / 1e9
        if "ready_ns" in result else None)
    result["first_content_s"] = (
        (result["first_content_ns"] - result["sent_ns"]) / 1e9
        if "first_content_ns" in result else None)
    result["decode_tok_s"] = (
        sum(len(e[1]) for e in events[1:]) / ((events[-1][0] - events[0][0]) / 1e9)
        if gaps and events[-1][0] > events[0][0] else None)
    result["stream_gap_s"] = {
        "p50": percentile(gaps, .5), "p95": percentile(gaps, .95),
        "p99": percentile(gaps, .99), "max": max(gaps, default=None),
        "over_500ms": sum(g > .5 for g in gaps),
        "over_1s": sum(g > 1 for g in gaps)}
    return result


class Run:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.origin = time.monotonic_ns()
        self.events = (self.root / "events.jsonl").open("x", buffering=1)
        self.results = []
        self.slots = asyncio.Semaphore(args.max_outstanding)

    def emit(self, event, **data):
        self.events.write(json.dumps({"event": event, "ns": time.monotonic_ns(),
                                      **data}) + "\n")

    async def request(self, session, body, name):
        result = {"name": name, "status": "pending", "tokens": [],
                  "message": {"role": "assistant", "content": ""},
                  "body_sha256": digest(body), "ready_ns": time.monotonic_ns()}
        self.results.append(result)
        (self.root / f"{name}.request.json").write_text(json.dumps(body))
        try:
            async with self.slots:
                result["sent_ns"] = time.monotonic_ns()
                self.emit("send", name=name)
                for attempt in range(3):
                    async with session.post(self.args.url + "/v1/chat/completions",
                                            json=body) as response:
                        result["http_status"] = response.status
                        self.emit("headers", name=name, status=response.status,
                                  attempt=attempt)
                        if response.status in (429, 503) and attempt < 2:
                            await response.read()
                            await asyncio.sleep(1 + attempt)
                            continue
                        if response.status != 200:
                            raise RuntimeError(f"HTTP {response.status}: "
                                               + (await response.text())[:1000])
                        done = False
                        async for payload in sse(response.content):
                            now = time.monotonic_ns()
                            if payload == "[DONE]":
                                done = True
                                break
                            chunk = json.loads(payload)
                            if chunk.get("error"):
                                raise RuntimeError(str(chunk["error"]))
                            for key in ("usage", "metrics"):
                                if chunk.get(key) is not None:
                                    result[key] = chunk[key]
                            for choice in chunk.get("choices", []):
                                ids = choice.get("token_ids") or []
                                if ids:
                                    result["tokens"].append([now, ids])
                                    self.emit("tokens", name=name, count=len(ids))
                                delta = choice.get("delta") or {}
                                if delta.get("content"):
                                    result.setdefault("first_content_ns", now)
                                if delta.get("reasoning") or delta.get("reasoning_content"):
                                    result.setdefault("first_reasoning_ns", now)
                                merge_delta(result["message"], delta)
                                if choice.get("finish_reason"):
                                    result["finish_reason"] = choice["finish_reason"]
                        if not done or not result.get("finish_reason"):
                            raise RuntimeError("Truncated SSE response")
                        if not result["tokens"]:
                            raise RuntimeError("No exact token IDs in stream")
                        result["status"] = "complete"
                        break
        except asyncio.CancelledError:
            result["status"] = "cancelled_deadline_or_signal"
            raise
        except Exception as exc:
            result["status"], result["error"] = "failed", str(exc)
        finally:
            result["finished_ns"] = time.monotonic_ns()
            if "sent_ns" in result:
                summarize(result)
            self.emit("finished", name=name, status=result["status"])
            (self.root / f"{name}.result.json").write_text(json.dumps(result))
            print(json.dumps({k: result.get(k) for k in (
                "name", "status", "ttft_s", "decode_tok_s", "stream_gap_s",
                "usage", "token_accounting_delta", "error")}), flush=True)
        return result

    async def conversation(self, session, spec):
        await asyncio.sleep(max(0, spec["arrival_s"] -
                                (time.monotonic_ns() - self.origin) / 1e9))
        messages = [{"role": "user", "content": spec["prefix"] +
                     self.manifest["corpus"] + "\n" + self.manifest["question"]}]
        for turn, count in enumerate(self.manifest["output_budgets"]):
            name = f"c{spec['id']}-t{turn}"
            body = {**self.manifest["sampling"], "model": self.args.model,
                    "request_id": name, "messages": messages, "tools": TOOLS,
                    "tool_choice": "auto", "max_tokens": count,
                    "stream": True, "return_token_ids": True,
                    "stream_options": {"include_usage": True},
                    "seed": 1234 + spec["id"] * 10 + turn}
            result = await self.request(session, body, name)
            if result["status"] != "complete":
                return
            if turn + 1 == len(self.manifest["output_budgets"]):
                return
            messages.append(result["message"])
            self.emit("tool_or_user_wait", name=name, delay_s=spec["delay_s"])
            await asyncio.sleep(spec["delay_s"])
            calls = result["message"].get("tool_calls") or []
            if calls:
                for call in calls:
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": self.manifest["tool_result"]})
            else:
                messages.append({"role": "user", "content":
                    self.manifest["tool_followup"] if turn == 0 and
                    self.manifest.get("tool_followup") else
                    "Here is the requested file/diagnostic output:\n" +
                    self.manifest["tool_result"] +
                    "\nContinue with a detailed analysis and concrete implementation."})

    async def metrics(self, session):
        while True:
            try:
                nginx_down()
                async with session.get(self.args.url + "/metrics",
                                       timeout=aiohttp.ClientTimeout(total=5)) as r:
                    self.emit("metrics", status=r.status, text=await r.text())
                self.emit("system", **system_snapshot())
            except RuntimeError:
                raise
            except Exception as exc:
                self.emit("metrics_error", error=str(exc))
            await asyncio.sleep(2)

    async def replay(self, session, entry, origin, directory):
        await asyncio.sleep(max(0, (entry["sent_ns"] - origin) / 1e9 -
                                (time.monotonic_ns() - self.origin) / 1e9))
        body = json.loads((directory / f"{entry['name']}.request.json").read_text())
        return await self.request(session, body, entry["name"])

    async def run(self):
        nginx_down()
        headers = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        gate = Path(self.args.trace_gate) if self.args.trace_gate else None
        meta = {"manifest_sha256": digest(self.manifest), "origin_ns": self.origin,
                "wall_time_ns": time.time_ns(), "args": vars(self.args),
                "configuration": configuration_snapshot(),
                "git": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               text=True).strip()}
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        tasks = []
        if gate:
            gate.touch(exist_ok=False)
        try:
            # return_token_ids includes the entire 40k prompt in the first SSE
            # event. aiohttp's default 128 KiB line cap is too small for it.
            async with aiohttp.ClientSession(headers=headers, read_bufsize=4*1024*1024,
                    timeout=
                    aiohttp.ClientTimeout(total=self.args.request_timeout)) as session:
                async with asyncio.timeout(self.args.duration):
                    async with asyncio.TaskGroup() as group:
                        monitor = group.create_task(self.metrics(session))
                        if self.args.replay_from:
                            directory = Path(self.args.replay_from)
                            source_meta = json.loads((directory / "meta.json").read_text())
                            if source_meta["manifest_sha256"] != digest(self.manifest):
                                raise ValueError("Replay manifest differs from source")
                            entries = json.loads((directory / "summary.json").read_text())
                            tasks = [group.create_task(self.replay(session, entry,
                                source_meta["origin_ns"], directory))
                                for entry in entries if "sent_ns" in entry]
                        else:
                            tasks = [group.create_task(self.conversation(session, spec))
                                     for spec in self.manifest["conversations"]]
                        await asyncio.gather(*tasks)
                        self.emit("cache_drain_begin")
                        await asyncio.sleep(self.args.drain_seconds)
                        monitor.cancel()
        except TimeoutError:
            self.emit("deadline")
        finally:
            if gate:
                gate.unlink(missing_ok=True)
            meta["finished_ns"] = time.monotonic_ns()
            (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
            planned = [f"c{s['id']}-t{t}" for s in self.manifest["conversations"]
                       for t in range(len(self.manifest["output_budgets"]))]
            seen = {r["name"] for r in self.results}
            self.results.extend({"name": name, "status": "not_started_dependency_or_deadline"}
                                for name in planned if name not in seen)
            compact = [{k: v for k, v in r.items() if k not in ("tokens", "message")}
                       for r in self.results]
            (self.root / "summary.json").write_text(json.dumps(compact, indent=2))
            self.events.close()
        return any(r["status"] != "complete" for r in self.results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--context", type=int, default=40000)
    parser.add_argument("--prompt-namespace", default="frozen-mixed-v1")
    parser.add_argument("--output")
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument("--model", default="qwen3.8-27b-gptq8")
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=480)
    parser.add_argument("--max-outstanding", type=int, choices=range(1, 33), default=8)
    parser.add_argument("--trace-gate")
    parser.add_argument("--drain-seconds", type=float, default=5)
    parser.add_argument("--replay-from", help="Exact saved bodies and send offsets; "
                        "open-loop replay, not response-dependent agent mode")
    args = parser.parse_args()
    if args.prepare:
        from garble_docs_probe import build_corpus
        from garble_repro2 import get_tok
        corpus = build_corpus(get_tok(), args.context)
        manifest = {"version": 1, "corpus": corpus, "context_target": args.context,
                    "sampling": {"temperature": 1, "top_p": .95, "top_k": 20,
                                 "reasoning_effort": "low",
                                 "chat_template_kwargs": {"enable_thinking": True}},
                    "output_budgets": [1536, 256, 768],
                    "tool_followup": "Before continuing, call read_file with path "
                    "README.md to inspect the actual project instructions. Do not "
                    "write the final analysis until the tool result is returned.",
                    "question": "Write exhaustive technical release notes for this "
                    "system. Discuss correctness and performance in depth. Use the "
                    "provided documents; read_file is available for missing details.",
                    "tool_result": "Diagnostic: 40k context, one active decode at "
                    "20 tokens/s. New long prompts cause visible stream pauses. "
                    "INT8 weights/KV and fp32 recurrent state must remain enabled.\n"
                    + corpus[-8000:],
                    "conversations": [{"id": i, "arrival_s": t, "delay_s": 2+i%3*3,
                        "prefix": f"Independent review {args.prompt_namespace}, session {i}.\n"}
                        for i, t in enumerate([0, 7, 19, 29, 43, 61, 83, 109])]}
        with open(args.manifest, "x") as f:
            json.dump(manifest, f)
        print(digest(manifest))
        return
    if not args.output:
        parser.error("--output is required for a run")
    if args.duration <= 0 or args.request_timeout <= 0 or args.drain_seconds < 0:
        parser.error("timeouts must be positive")
    manifest = json.loads(Path(args.manifest).read_text())
    async def start():
        current = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, current.cancel)
        return await Run(args, manifest).run()
    raise SystemExit(2 if asyncio.run(start()) else 0)


if __name__ == "__main__":
    main()
