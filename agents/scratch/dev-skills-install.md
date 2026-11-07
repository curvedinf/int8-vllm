# dev-skills installation record

- **Source:** https://github.com/curvedinf/dev-skills @ `d1754529343d4c7bc8fb6de40565e80757f0dbaa` (main, 2026-10-14, "improved progressive disclosure of triggered events")
- **Scope:** project-wide (user specified this repository as the destination)
- **Destination mapping:** `skills/{activated,passive,triggered}/` → `agents/skills/{activated,passive,triggered}/`, byte-identical (verified with `diff -r`); `LICENSE` → `agents/skills/LICENSE` (MIT, curvedinf)
- **Not copied:** upstream `README.md` and `docs/skill-research.md` (collection marketing/research notes, not project-operational); `skills/.gitkeep`; upstream `AGENTS.md` merged by topic into the project `AGENTS.md` instead of being copied
- **AGENTS.md:** added the "Agent skills (dev-skills)" section between "What `done` means here" and the upstream appendix — request-term routing, scoped passive defaults, scratch policy, pre-commit security gate, self-improvement, authoring conventions, reference libraries. Recorded that project policy overrides generic skill guidance (testing ladder, push-to-`main`, A/B-ledger doctrine)
- **.gitignore:** appended `/agents/tmp/`; no unrelated rules altered
- **timers.json:** initialized `maintain-scratch` from the declared default (7-day interval, `null` completion); no upstream completion history imported; no legacy `maintenance.json` existed to migrate
- **security-policy.md:** created as a project-local template with no approved permissions; the upstream project's approved disclosure permission was deliberately **not** imported (installer rule: never import upstream disclosure permissions)
- **Adapters:** `agents/openai.yaml` copied per skill (implicit invocation disabled for activated skills, enabled for passive/triggered)
- **Collisions:** none — no same-name skills existed; `.claude/skills/ci-fails-buildkite` is untouched and unaffected
- **Self-improvement:** enabled.
- **Loading:** skills load via the explicit links in `AGENTS.md`, not host-native discovery; new sessions pick up the merged `AGENTS.md` automatically (current sessions must re-read it)
- **Reinstall:** idempotent — re-run install-dev-skills and merge against this revision
- **Local customization:** `refine-skill` patched; preserve on reinstall

## Update 2026-11-07 — dev-skills @ a8623a537af1d3ae989f0091c75a41ed78a2253f

- **Source:** https://github.com/curvedinf/dev-skills @ `a8623a537af1d3ae989f0091c75a41ed78a2253f` (main, 2026-11-07, "context compaction procedure"); prior revision `d1754529...`
- **Added:** `context-paging` (activated; full folder incl. scripts/, references/kimi-hooks.md, adapters) and `context-compaction-procedure` (triggered) — byte-identical to upstream (`diff -r` clean)
- **Updated:** install-dev-skills/SKILL.md replaced with the new upstream revision (verified local == old upstream first — no local edits lost); other skills untouched, local `refine-skill` patch preserved (upstream did not change it)
- **Not imported:** upstream AGENTS.md copy, README.md/docs, upstream timers completion history (timers.json unchanged: `maintain-scratch` last_completed_at stays null)
- **AGENTS.md merged:** routing rows for context-paging + context-compaction-procedure; new sections "Context paging" (opt-in check procedure, project paths adapted to `agents/skills/...`) and "Context compaction" (fresh-session read directive) inserted after the Scratch space section
- **.gitignore:** added `/agents/context-paging/` (upstream `__pycache__/` already covered)
- **Context-paging harness (user opted in 2026-11-07):** dispatcher installed at `~/.kimi-code/hooks/context-paging/hook.py` from the reference template; three `[[hooks]]` rules appended to `~/.kimi-code/config.toml` (SessionStart startup|resume, UserPromptSubmit, SessionEnd exit|archive; timeout 15; validated with `kimi doctor config`, timestamped backup kept); `cp_subscribe.py` run from the project root — registry `~/.config/context-paging/projects.json` maps `/home/curved/int8-vllm` to `agents/context-paging/context.db`
- **Verified:** manual stdin dispatcher test captured a UserPromptSubmit message in the subscribed project and printed the SessionStart paging notice; an unsubscribed cwd produced no capture and exited 0 (fail-open). `/reload` now (or the next session) activates the hooks.
- **Self-improvement:** unchanged (enabled).
