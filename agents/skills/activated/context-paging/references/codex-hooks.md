# Codex CLI harness wiring

Second worked example for coding the capture plugin at install time, mirroring the [kimi wiring](kimi-hooks.md) architecture with one key difference: Codex rollouts flush incrementally, so a single sweep mechanism covers both mid-turn and turn-end capture. Verify events, payload fields, and the transcript format against the [Codex hooks documentation](https://developers.openai.com/codex/hooks) before writing anything; this page matches the documented API and the observed rollout format as of 2026-11-08 (codex-cli 0.167.0).

Facts this wiring relies on:

- Hooks are `hooks.json` (used here) or inline `[hooks.<Event>]` tables in `~/.codex/config.toml` — never both in one layer. Command hooks receive one JSON object on stdin with `session_id`, `transcript_path`, `cwd`, `hook_event_name`, and `model`.
- `UserPromptSubmit` carries `prompt` as a plain string. `Stop` and `SubagentStop` carry `last_assistant_message`, but only the final reply of a turn; mid-turn assistant prose between tool calls is not in any event payload.
- `transcript_path` (main thread) and `agent_transcript_path` (subagents, on `SubagentStop`) point at the session's rollout transcript: one JSON object per line, flushed incrementally as each item completes (verified: tool calls and outputs land with distinct timestamps mid-turn), with `response_item` records carrying payloads `message` (role user/assistant/developer — developer messages become `system` rows), `reasoning` (raw thinking encrypted; cleartext `summary` items occasionally present), `function_call`/`custom_tool_call` (name + arguments/input), and `function_call_output`/`custom_tool_call_output` (output). Codex documents the transcript format as unstable — parse defensively and re-verify after upgrades. This differs from kimi code CLI, whose wire transcript batches at turn boundaries and therefore cannot serve mid-turn sweeps; that is why the kimi example inserts tool rows from `PostToolUse` payloads instead.
- `SessionEnd` fires on archive, delete, normal close, or ~30 minutes idle; its timeout caps at 3 seconds, so its sweep must stay small (usually it is — earlier events already swept).
- Plain text on stdout is invalid for `Stop` and `SubagentStop`; print nothing on turn-end events. Only the `SessionStart` notice uses stdout.
- Codex gates non-managed hooks on trust: changed hook definitions are skipped until reviewed with `/hooks` in the CLI. Re-review after any edit to the command or script, and expect a startup warning until then.

Capture model: `user` rows insert live at `UserPromptSubmit`; `PostToolUse`, `Stop`, `SubagentStop`, and `SessionEnd` sweep the rollout transcript(s) past a cursor stored in `agents/context-paging/cursors.json`, inserting `thinking` (cleartext reasoning summaries), `assistant`, `tool_call`, `tool`, and `system` (developer messages) rows in exact conversation order, plus `user` rows for subagent prompts from the subagent transcript. Because rollouts flush incrementally, the `PostToolUse` sweep lands each tool call and its result right after it executes — live capture from a single source, with no payload inserts and therefore no duplication risk. Sweeps are idempotent and chunked (200 records per transcript per event); a file lock serializes concurrent sweeps.

## Steps

1. Write the dispatcher to `~/.codex/hooks/context-paging/hook.py` from the template below.
2. Register the six events in `~/.codex/hooks.json` (below).
3. Open codex and review the changed hooks in `/hooks` so they are trusted.
4. From the project root, run `cp_subscribe.py` — last, after the rules exist and are trusted, because subscription means "installed and activated".
5. Verify (below) in a fresh session.

## Dispatcher template

```python
#!/usr/bin/env python3
"""Codex context-paging hook: per-project gate, lifecycle + full-turn capture.

Captures user prompts live (UserPromptSubmit) and sweeps the session rollout
transcript after every tool call and at turn end (PostToolUse, Stop,
SubagentStop, SessionEnd) so agent thinking summaries, replies, tool calls,
and tool results are recorded verbatim and in order. A cursor file next to
the database tracks how far each transcript has been swept, making capture
idempotent across repeated events.
"""
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
MAX_RECORDS = 200  # per transcript per event; leftovers sweep on the next event
SWEEP_EVENTS = ("PostToolUse", "Stop", "SubagentStop", "SessionEnd")


def run(entry, root, sid, name, *args, stdin_text=None, timeout=10):
    env = dict(os.environ, CP_SESSION=sid, CP_PROJECT=root,
               CP_DB=entry.get("db", ""), CP_HARNESS="codex")
    try:
        subprocess.run(
            [sys.executable or "python3", os.path.join(entry.get("scripts", ""), name), *args],
            cwd=root, input=stdin_text, text=True, env=env,
            stdout=subprocess.DEVNULL, stderr=None, timeout=timeout,
        )
    except Exception as exc:  # fail-open: capture must never block
        print("context-paging: %s failed: %s" % (name, exc), file=sys.stderr)


def block_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def rollout_rows(payload, capture_user):
    """Yield (role, content) pairs for one response_item payload."""
    ptype = payload.get("type")
    if ptype == "message":
        role = payload.get("role", "")
        text = block_text(payload.get("content"))
        if not text.strip():
            return
        if role == "assistant":
            yield "assistant", text
        elif role == "user" and capture_user:
            # subagent prompts (main prompts arrive via UserPromptSubmit instead)
            yield "user", text
        elif role == "developer":
            yield "system", text
    elif ptype == "reasoning":
        # raw thinking is encrypted; capture cleartext summaries when present
        text = block_text(payload.get("summary"))
        if text.strip():
            yield "thinking", text
    elif ptype in ("function_call", "custom_tool_call"):
        yield "tool_call", "%s %s" % (
            payload.get("name", "?"),
            payload.get("arguments") or payload.get("input") or "",
        )
    elif ptype in ("function_call_output", "custom_tool_call_output"):
        text = block_text(payload.get("output"))
        if text.strip():
            yield "tool", text


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


def sweep(entry, root, sid, transcript, capture_user):
    """Insert message rows for rollout records past the cursor."""
    if not transcript:
        return
    try:
        with open(transcript, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    cur_path, cursors = load_cursors(entry.get("db", ""))
    lock = acquire_lock(os.path.dirname(cur_path))
    if lock is None:
        return  # another sweep holds the lock; the next event catches up
    try:
        done = cursors.get(transcript, 0)
        if len(lines) < done:  # transcript replaced: resweep from the start
            done = 0
        examined = 0
        for line in lines[done:done + MAX_RECORDS]:
            try:
                rec = json.loads(line)
            except ValueError:
                break  # partial line mid-write: retry at the next event
            examined += 1
            if rec.get("type") != "response_item":
                continue
            for role, content in rollout_rows(rec.get("payload") or {}, capture_user):
                run(entry, root, sid, "cp_message_add.py", role, "-", stdin_text=content)
        cursors[transcript] = done + examined
        save_cursors(cur_path, cursors)
    finally:
        if lock:
            lock.close()


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
    sid, event = payload.get("session_id", ""), payload.get("hook_event_name", "")
    if event == "SessionStart":
        run(entry, root, sid, "cp_session_start.py", sid)
        run(entry, root, sid, "cp_cleanup.py")
        print(
            "Context paging active for this project. Nothing is summarized; the "
            "full conversation (prompts, agent thinking, replies, tool calls "
            "and results) is captured to %s. Recover it: "
            "`python3 %s/cp_search.py <query>` or cp_page_in.py; store exact "
            "durable detail with cp_page_out.py." % (entry.get("db", ""), entry.get("scripts", ""))
        )
    elif event == "UserPromptSubmit":
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            run(entry, root, sid, "cp_message_add.py", "user", "-", stdin_text=prompt)
    elif event in SWEEP_EVENTS:
        if event == "SubagentStop":
            sweep(entry, root, sid, payload.get("agent_transcript_path"), capture_user=True)
        else:
            sweep(entry, root, sid, payload.get("transcript_path"), capture_user=False)
        if event == "SessionEnd":
            run(entry, root, sid, "cp_session_end.py", sid, timeout=2)


if __name__ == "__main__":
    main()
    sys.exit(0)
```

## Hook rules (`~/.codex/hooks.json`)

```json
{
  "description": "Context paging capture for subscribed projects (see ~/.config/context-paging/projects.json)",
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "python3 ~/.codex/hooks/context-paging/hook.py", "timeout": 15}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python3 ~/.codex/hooks/context-paging/hook.py", "timeout": 15}]}],
    "PostToolUse": [{"hooks": [{"type": "command", "command": "python3 ~/.codex/hooks/context-paging/hook.py", "timeout": 15}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "python3 ~/.codex/hooks/context-paging/hook.py", "timeout": 30}]}],
    "SubagentStop": [{"hooks": [{"type": "command", "command": "python3 ~/.codex/hooks/context-paging/hook.py", "timeout": 30}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "python3 ~/.codex/hooks/context-paging/hook.py", "timeout": 3}]}]
  }
}
```

Codex expands `~` in hook commands; use an absolute path if your build does not.

## Practices

- **Gate first, fail open, stay idempotent** — identical to the kimi wiring: registry read before any database access, stderr-only error paths with exit 0, cursor advancing only past examined lines.
- **One sweep source.** Rollouts flush incrementally, so every row — tools included — comes from the sweep; no event-payload inserts exist to duplicate it. The `PostToolUse` trigger is what makes capture land right after each tool call.
- **Serialize sweeps.** Concurrent events take the non-blocking file lock; the loser skips and the next event catches up.
- **Respect event output contracts.** Print nothing on `Stop`, `SubagentStop`, and `SessionEnd`; plain stdout there is invalid. The `SessionStart` notice is the only stdout this dispatcher produces.
- **Sweep the documented paths.** `transcript_path` and `agent_transcript_path` come from the payload; never guess rollout file locations. `SessionEnd` gets at most 3 seconds — its sweep is a final backstop, not the primary capture.
- **Trust changes.** Codex skips changed hooks until `/hooks` review; re-verify after any edit to the dispatcher or rules.
- **Unstable transcript format.** The rollout schema is not a stable interface. The parser tolerates unknown payload types and fails open; after a codex upgrade, re-run the verify steps and adjust `rollout_rows` if record shapes changed.

## Verify

1. Review the hooks in `/hooks` (codex skips them until trusted).
2. Start a session in the subscribed project: `cp_session_list.py` shows a new active `codex` session and the paging notice appears.
3. Send one prompt that runs a shell command: `cp_message_list.py --since any` (with `CP_SESSION` set to the session id) shows the `user` prompt, the `tool_call` row, the `tool` output, and the `assistant` reply, in order — and the tool rows are present as soon as the command finishes, not only at turn end.
4. Run a session in a different, unsubscribed project: no rows appear anywhere and the hook still exits 0.
5. Manual harness check without starting a session:
   `echo '{"hook_event_name":"UserPromptSubmit","session_id":"t1","cwd":"<project root>","prompt":"hello"}' | python3 ~/.codex/hooks/context-paging/hook.py`

## Remove

Run `cp_unsubscribe.py` in each subscribed project, delete the six events from `~/.codex/hooks.json` and `~/.codex/hooks/context-paging/`, then review the removal in `/hooks`. Databases under `agents/context-paging/` stay until deleted.
