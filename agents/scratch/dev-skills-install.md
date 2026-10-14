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
