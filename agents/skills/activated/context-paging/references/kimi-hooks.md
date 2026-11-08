# Kimi Code CLI harness wiring

Worked example for coding the capture plugin at install time. Verify event names, payload fields, and config syntax against the [hooks documentation](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/hooks.html) before writing anything; this page matches the documented API and the observed `wire.jsonl` format as of 2026-11-08 (kimi 2.5.2).

Facts this wiring relies on:

- Hook rules are `[[hooks]]` entries in `~/.kimi-code/config.toml` (user-level only; there is no project hook file) with fields `event`, `matcher`, `command`, `timeout` (1–600s, default 30).
- Commands run with the session's local project directory as working directory; event details arrive as JSON on stdin, including `hook_event_name`, `session_id`, and `cwd`.
- `UserPromptSubmit`'s `prompt` field may be a string or a list of content blocks (`[{"type": "text", "text": "..."}]`); the dispatcher normalizes it to text before capture.
- `PostToolUse` fires after every tool call — including subagent tool calls — carrying `tool_name`, `tool_input` (object), and `tool_output` (string); `PostToolUseFailure` carries `error.message` instead. These payloads are the only mid-turn capture source: the wire transcript is written in batches at turn boundaries, so it is empty mid-turn no matter which event sweeps it.
- `Stop` fires when the model ends its turn but its payload carries only `stop_hook_active` — no assistant text, and no event carries thinking at all. Both live in the session's internal transcript `~/.kimi-code/sessions/<wd>/session_<id>/agents/<agent>/wire.jsonl`: one JSON object per line, with `agent.message.appended` records nesting `message.message` as `{role, content, toolCalls}` (assistant content blocks have `type` `"text"` key `text` or `"think"` key `think`; `user` records with a non-`user` `meta.origin.kind` are system-triggered inputs and task notifications) and `context.append_message` records carrying injected reminders, task prompts, and skill activations in `message.origin.kind`. This format is undocumented and may change between versions; the dispatcher below parses it defensively and fails open.
- `SubagentStop` fires after the subagent's transcript is flushed, so its sweep sees the subagent's full turn; subagent prompts are `user`-role records in the subagent's own transcript.
- Exit 0 allows; stdout may be appended to context (only used for the session-start notice); failures fail open.

Capture model: `user` rows insert live at `UserPromptSubmit`; `tool_call` (name + input JSON) and `tool` (verbatim output, or the failure message) rows insert live from `PostToolUse`/`PostToolUseFailure` — main and subagent tools alike; `Stop`, `SubagentStop`, and `SessionEnd` sweep each `wire.jsonl` past a cursor stored in `agents/context-paging/cursors.json`, inserting `thinking` (think blocks), `assistant` (replies), subagent `user` (prompts), and `system` rows (injected reminders, system-triggered inputs, task notifications). The sweep never emits tool rows — events already inserted them — so no event or sweep ever duplicates a row. Sweeps are idempotent and chunked (200 records per transcript per event); a file lock serializes concurrent sweeps (parallel tool calls fire events concurrently). Because thinking, replies, and system rows sweep at turn end, they carry end-of-turn timestamps; content is exact, intra-turn ordering is approximate.

## Steps

1. Write the dispatcher to `~/.kimi-code/hooks/context-paging/hook.py` from the template below.
2. Append the seven hook rules to `~/.kimi-code/config.toml`.
3. From the project root, run `cp_subscribe.py` — last, after the rules exist, because subscription means "installed and activated".
4. Verify (below) in a fresh session.

## Dispatcher template

```python
#!/usr/bin/env python3
"""kimi code context-paging hook: per-project gate, lifecycle + full-turn capture.

User prompts insert live (UserPromptSubmit); tool calls and results insert
live from PostToolUse/PostToolUseFailure payloads (the wire transcript is
batched at turn end, so it cannot serve mid-turn); Stop/SubagentStop/
SessionEnd sweep the wire.jsonl transcripts for agent thinking and replies,
which no event payload carries. A cursor file next to the database tracks
how far each transcript has been swept, making sweeps idempotent.
"""
import glob
import json
import os
import subprocess
import sys

try:
    import fcntl
except ImportError:  # locking unavailable (non-POSIX): sweeps proceed unlocked
    fcntl = None

REGISTRY = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "context-paging", "projects.json",
)
SESSIONS_DIR = os.path.join(os.path.expanduser("~"), ".kimi-code", "sessions")
MAX_RECORDS = 200  # per transcript per event; leftovers sweep on the next event


def run(scripts, name, *args, cwd, sid="", stdin_text=None, timeout=10):
    try:
        subprocess.run(
            [sys.executable or "python3", os.path.join(scripts, name), *args],
            cwd=cwd, input=stdin_text, text=True,
            env=dict(os.environ, CP_SESSION=sid, CP_HARNESS="kimi-code"),
            stdout=subprocess.DEVNULL, stderr=None, timeout=timeout,
        )
    except Exception as exc:  # fail-open: capture must never block
        print("context-paging: %s failed: %s" % (name, exc), file=sys.stderr)


def prompt_text(payload):
    p = payload.get("prompt", "")
    if isinstance(p, list):  # content blocks: [{"type": "text", "text": "..."}]
        return "\n".join(b.get("text", "") for b in p if isinstance(b, dict))
    return p if isinstance(p, str) else ""


def output_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(b.get("text", "") for b in value if isinstance(b, dict))
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def capture_tool_use(scripts, root, sid, payload, failure=False):
    """Insert tool_call + tool rows live from a PostToolUse( Failure) payload."""
    name = payload.get("tool_name", "?")
    tool_input = payload.get("tool_input")
    call = "%s %s" % (name, tool_input if isinstance(tool_input, str)
                      else json.dumps(tool_input or {}, sort_keys=True))
    if failure:
        error = payload.get("error") or {}
        result = output_text(error.get("message") or error)
    else:
        result = output_text(payload.get("tool_output"))
    run(scripts, "cp_message_add.py", "tool_call", "-", cwd=root, sid=sid,
        stdin_text=call)
    if result.strip():
        run(scripts, "cp_message_add.py", "tool", "-", cwd=root, sid=sid,
            stdin_text=result)


def typed_text(content, block_type, key):
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b.get(key, "") for b in content
        if isinstance(b, dict) and b.get("type") == block_type
    )


def wire_rows(rec, capture_user):
    """Yield (role, content) pairs for one wire record.

    Tool calls and results arrive live via PostToolUse events (also for
    subagents), so the sweep takes only thinking, replies, subagent
    prompts, and harness-side system content — never tool rows, which
    would duplicate the event inserts.
    """
    rtype = rec.get("type")
    if rtype == "agent.message.appended":
        outer = rec.get("message") or {}
        msg = outer.get("message") or {}
        origin = (outer.get("meta") or {}).get("origin") or {}
        kind = origin.get("kind")
        if msg.get("role") == "assistant":
            thinking = typed_text(msg.get("content"), "think", "think")
            if thinking.strip():
                yield "thinking", thinking
            text = typed_text(msg.get("content"), "text", "text")
            if text.strip():
                yield "assistant", text
        elif msg.get("role") == "user":
            if kind == "user":
                # main prompts arrive via UserPromptSubmit; subagent prompts
                # arrive only here, in the subagent's own transcript
                if capture_user:
                    text = typed_text(msg.get("content"), "text", "text")
                    if text.strip():
                        yield "user", text
            else:  # system_trigger inputs, task notifications, and the like
                text = typed_text(msg.get("content"), "text", "text")
                if text.strip():
                    yield "system", text
    elif rtype == "context.append_message":
        msg = rec.get("message") or {}
        # injected reminders, task prompts, and skill activations appear only
        # in this stream; user prompts and system_trigger inputs are already
        # carried by agent.message.appended, so skip them here
        kind = (msg.get("origin") or {}).get("kind")
        if kind not in (None, "user", "system_trigger"):
            text = typed_text(msg.get("content"), "text", "text")
            if text.strip():
                yield "system", text


def load_cursors(db):
    path = os.path.join(os.path.dirname(db), "cursors.json")
    try:
        with open(path, encoding="utf-8") as f:
            return path, json.load(f)
    except Exception:
        return path, {}


def save_cursors(path, cursors):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cursors, f, indent=1, sort_keys=True)
    except Exception as exc:
        print("context-paging: cursor save failed: %s" % exc, file=sys.stderr)


def acquire_lock(dirpath):
    """Return a locked file object, None if another sweep holds it, or False
    when locking is unavailable (proceed unlocked)."""
    if fcntl is None:
        return False
    try:
        os.makedirs(dirpath, exist_ok=True)
        fd = open(os.path.join(dirpath, "cursors.lock"), "w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        return None


def sweep(scripts, db, sid, cwd, wire_path, capture_user):
    """Insert message rows for wire.jsonl records past the cursor."""
    try:
        with open(wire_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    cur_path, cursors = load_cursors(db)
    lock = acquire_lock(os.path.dirname(cur_path))
    if lock is None:
        return  # another sweep holds the lock; the next event catches up
    try:
        done = cursors.get(wire_path, 0)
        if len(lines) < done:  # transcript replaced: resweep from the start
            done = 0
        examined = 0
        for line in lines[done:done + MAX_RECORDS]:
            try:
                rec = json.loads(line)
            except ValueError:
                break  # partial line mid-write: retry at the next event
            examined += 1
            if rec.get("type") not in ("agent.message.appended", "context.append_message"):
                continue
            for role, content in wire_rows(rec, capture_user):
                run(scripts, "cp_message_add.py", role, "-", cwd=cwd, sid=sid,
                    stdin_text=content)
        cursors[wire_path] = done + examined
        save_cursors(cur_path, cursors)
    finally:
        if lock:
            lock.close()


def session_wire_paths(sid):
    return glob.glob(os.path.join(SESSIONS_DIR, "*", sid, "agents", "*", "wire.jsonl"))


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    root = payload.get("cwd") or os.getcwd()
    try:
        with open(REGISTRY, encoding="utf-8") as f:
            entry = json.load(f).get(root)
    except Exception:
        entry = None
    if not entry:
        return  # not subscribed: this project never runs capture
    scripts, db = entry.get("scripts", ""), entry.get("db", "")
    sid, event = payload.get("session_id", ""), payload.get("hook_event_name", "")
    if event == "SessionStart":
        run(scripts, "cp_session_start.py", sid, cwd=root, sid=sid)
        run(scripts, "cp_cleanup.py", cwd=root, sid=sid)
        print(
            "Context paging active for this project. Nothing is summarized; the "
            "full conversation (prompts, agent thinking, replies, tool calls "
            "and results) is captured to %s. Recover it: "
            "`python3 %s/cp_search.py <query>` or cp_page_in.py; store exact "
            "durable detail with cp_page_out.py." % (db, scripts)
        )
    elif event == "UserPromptSubmit":
        run(scripts, "cp_message_add.py", "user", "-", cwd=root, sid=sid,
            stdin_text=prompt_text(payload))
    elif event in ("PostToolUse", "PostToolUseFailure"):
        capture_tool_use(scripts, root, sid, payload,
                         failure=(event == "PostToolUseFailure"))
    elif event in ("Stop", "SubagentStop", "SessionEnd"):
        for wire in session_wire_paths(sid):
            sweep(scripts, db, sid, root, wire, capture_user=os.path.basename(os.path.dirname(wire)) != "main")
        if event == "SessionEnd":
            run(scripts, "cp_session_end.py", sid, cwd=root, sid=sid)


if __name__ == "__main__":
    main()
    sys.exit(0)
```

## Hook rules

```toml
[[hooks]]
event = "SessionStart"
matcher = "startup|resume"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 15

[[hooks]]
event = "UserPromptSubmit"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 15

[[hooks]]
event = "PostToolUse"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 15

[[hooks]]
event = "PostToolUseFailure"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 15

[[hooks]]
event = "Stop"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 30

[[hooks]]
event = "SubagentStop"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 30

[[hooks]]
event = "SessionEnd"
matcher = "exit|archive"
command = "python3 ~/.kimi-code/hooks/context-paging/hook.py"
timeout = 15
```

## Practices

- **Gate first.** The registry read happens before any database access; an unsubscribed project costs one JSON read and exits 0. This is what keeps user-level hooks per-project.
- **Fail open.** Every error path logs to stderr and exits 0; never exit 2 — capture must not block a prompt, a tool call, or a turn.
- **One source per row.** Tool rows come only from `PostToolUse` payloads; thinking, replies, and subagent prompts come only from transcript sweeps. Neither source ever emits the other's rows, so repeated events and overlapping sweeps cannot duplicate anything.
- **Serialize sweeps.** Parallel tool calls fire hook events concurrently; the sweep takes a non-blocking file lock and skips if another sweep holds it — the next event catches up.
- **Keep semantics idempotent.** `cp_session_start.py` get-or-creates by session key, so `startup` and `resume` are both safe; `cp_cleanup.py` is due-checked; the cursor advances only past lines actually examined, so a partially written line is retried next event.
- **Undocumented format, defensive parse.** `wire.jsonl` is internal to the CLI. The dispatcher tolerates unknown record types and shapes and fails open; if a CLI update changes the format, capture degrades to the event-driven rows (prompts and tool activity). Re-verify the sweep after CLI upgrades.
- **Moved skill directory?** Rerun `cp_subscribe.py` from the project; the registry stores script paths.

## Verify

1. Start a fresh session in the subscribed project: `cp_session_list.py` shows a new active session, and the paging notice appears (SessionStart stdout may be appended to context).
2. Send one prompt that uses a tool, then a second tool that counts rows mid-turn (for example a short `python3 -c` against `agents/context-paging/context.db`): the count already includes the first tool's `tool_call` and `tool` rows — capture is live, not turn-end only.
3. After the turn ends, `cp_message_list.py --since any` (with `CP_SESSION` set to the session id) shows `user`, `tool_call`, `tool`, `thinking`, `assistant`, and `system` rows.
4. Ask for a subagent run: its tool rows appear exactly once (live events), its prompt, thinking, and reply arrive at `SubagentStop`, and the main thread's `Agent` call and result are captured.
5. Run a session in a different, unsubscribed project: no rows appear anywhere and the hook still exits 0.
6. Manual harness check without starting a session:
   `echo '{"hook_event_name":"UserPromptSubmit","session_id":"t1","cwd":"<project root>","prompt":[{"type":"text","text":"hello"}]}' | python3 ~/.kimi-code/hooks/context-paging/hook.py`

## Remove

Run `cp_unsubscribe.py` in each subscribed project, delete the seven `[[hooks]]` rules and `~/.kimi-code/hooks/context-paging/`. Databases under `agents/context-paging/` stay until deleted.
