# Oelala Loop — nightly development procedure (cold file)

> Executed by the daily reminder bound to the **oelala-loop Session** at **04:00 UTC**.
> This file is the authoritative procedure: the reminder only says "run the oelala
> loop", so a compacted or restored Session re-reads this document and continues here.
>
> Operator decisions baked in (2026-10-25): autonomy **everything except merge**;
> cadence **04:00 UTC daily** (same convention as the guardian loop); scope
> **this repository only**.

## 0. Environment (verified 2026-10-25)

| Fact | Value |
|---|---|
| Repository | `m0nklabs/oelala` (public; root `README.md` is a deliberate privacy decoy — never "fix" it) |
| Tracking issue | **#210** — "Oelala loop — nightly run log (04:00 UTC)"; the run report goes there |
| Loop worktree | `$HOME/oelala-loop` — branch `loop/workspace`, tracks `origin/main` |
| Operator clone | `$HOME/oelala` — **hands off**: it carries the operator's uncommitted work |
| Truth for code | `origin/main` (fetch first; the operator clone is often behind) |
| Test tooling | system `python3` has `fastapi` + `pytest`; `ruff` 0.15.15 on PATH. No project `.venv` — do not create a heavy one |
| GPU venv | `$HOME/venvs/gpu` is production-critical (`chattr +i`) — read-only use only, never delete/move/recreate |

Work only inside the loop worktree:

```bash
export WT="${OELALA_LOOP_WT:-$HOME/oelala-loop}"
cd "$WT" && git fetch origin --prune && git switch loop/workspace && git pull --ff-only
```

## 1. Scope and autonomy boundary

**In scope:** `m0nklabs/oelala` only — backend, frontend, workflows, compute
backends, docs, CI. Anything else (including `oelala-storage`, the guardian
gateway, the DSH harness) is out of scope, even when it looks related.

**Autonomy granted:**

- read every issue, PR, review thread and CI result in this repo;
- create branches, commit, push **branches**, open and update PRs;
- apply correct PR-Piet suggestions, reply on review threads with evidence;
- request PR-Piet reviews (`/review`, `/describe`, `/improve`, `/review2`) from the
  human `gh` account (`m0nk111`) after each push;
- open issues that describe verified defects or the next roadmap step;
- comment the nightly report on the tracking issue.

**Not granted (hard stops):**

1. **Never merge a PR**, never approve one, never dismiss a review, never resolve a
   thread that still has an unanswered finding. Leave the PR ready for the human merge.
2. **Never push to `main`** or any long-lived branch, never force-push, never rewrite
   published history, never delete branches or issues.
3. **Never restart or stop a systemd unit** (`oelala-backend`, `oelala-frontend`,
   `comfyui`, `minio`, and `frigate` especially). Validation only; a restart is a
   recommendation in the report.
4. **Never touch the operator clone** `$HOME/oelala` (dirty tree = their work), never
   stash/checkout/reset/clean it.
5. **Never touch `/home/flip/venvs`** (production runtime, `chattr +i`) and never run
   `npm run dev`, `uvicorn` or `python main.py` — services are systemd-only.
6. **Frigate is hands off** (security cameras, 24/7): read-only config, never "fix" it.
7. **Never touch another repository.** Cross-project findings are reported, not fixed.
8. **Never commit secrets**; never add private IPs, hostnames or environment-specific
   values to committed files (read them from `.env` / `compute_backends.json`).

## 2. Budget per night

Bounded so a run stays reviewable and cheap:

- **all** open-PR remediation (batched: one coherent tested push per PR head);
- **at most one** new implementation (one issue or one roadmap step), from a branch
  off `origin/main`;
- **at most one** pipeline-seeding action (§5) when PRs and issues are empty;
- no speculative refactors, dependency bumps or cleanup outside the task at hand.

A finished short run beats a long speculative one. "Nothing to do" is a valid,
reportable outcome.

## 3. Phase 0 — Preflight (evidence only, no writes)

```bash
cd "$WT" && git status --porcelain     # must be empty; dirty = report and stop
git fetch origin --prune && git log --oneline -5 origin/main
gh pr list -R m0nklabs/oelala --state open --json number,title,headRefName,isDraft,reviewDecision
gh issue list -R m0nklabs/oelala --state open --json number,title,labels,updatedAt
systemctl is-active oelala-backend oelala-frontend comfyui minio
```

- Dirty loop worktree → report and stop; never clean someone else's work.
- A service down → report it, but do **not** restart it (§1.3).
- Read `docs/HANDOFF.md` (current state) and `docs/ROADMAP.md` (next product step)
  on demand — cold files, not by habit in full.

## 4. Phase 1 — Open PRs (highest priority)

For each open PR, oldest-updated first:

```bash
gh pr view <N> -R m0nklabs/oelala --json reviews,comments,files,commits,statusCheckRollup,reviewDecision
gh api repos/m0nklabs/oelala/pulls/<N>/comments        # inline review threads
gh pr checks <N> -R m0nklabs/oelala
```

PR-Piet is wired through `.github/workflows/pr-piet.yml` → reusable
`m0nklabs/pr-piet/.github/workflows/reusable-pr-piet.yml`. It reviews on PR events
and on a `/review` comment from `m0nk111`; suggestions arrive as review bodies and
`suggestion` fences on inline threads.

For every finding in the latest PR-Piet review **at the current head SHA**:

1. Treat it as a hypothesis — verify against the code, docs and tests.
2. Correct and in scope → fix it in the same batch as the other findings for that head.
3. Wrong, already fixed, or out of scope → reply on the thread with the evidence
   (file:line, test output, or the commit that already covers it). Leave the thread
   open for the human.
4. Never leave a finding unaddressed and unanswered.

Then one coherent batch per PR:

```bash
python3 -m pytest tests/ -x -q          # or the targeted subset; GPU tests stay in tests/gpu
ruff check src/                          # plus ruff format --check on touched files
git commit … && git push                 # branch only
gh pr comment <N> -R m0nklabs/oelala --body '/review'
```

When a PR is settled (CI green, latest review covers the merge head, every finding
resolved or answered), leave one closing comment: what changed, the evidence, and
`Ready for human merge`. **Do not merge.** Draft PRs stay draft until validated.

## 5. Phase 2 — Open issues, then pipeline seeding

```bash
gh issue list -R m0nklabs/oelala --state open --json number,title,labels,updatedAt
```

- Triage: does it still reproduce? Is it already covered by a PR or a commit on
  `origin/main`? Stale → say so on the issue with evidence; **never close it**.
- Pick the highest-value *executable* issue that is not already in a PR; implement on
  `feat/<slug>-<issue>` or `fix/<slug>-<issue>` off `origin/main`, add tests that pin
  the changed contract, open the PR (`Closes #N`), then post `/review`.

**If no open PR or executable issue exists**, seed the pipeline from the unmerged
remote branches (state 2026-10-25, ahead/behind vs `origin/main`):

| Branch | Δ vs main | Last commit |
|---|---|---|
| `fix/prompt-gen-timeout-fallback` | +1 / −43 | 2026-09-24 |
| `feat/prompt-generator-h3-skill` | +1 / −46 | 2026-09-24 |
| `chore/remove-hardcoded-ips` | +1 / −74 | 2026-08-24 |
| `copilot/add-video-upscaling-api` | +5 / −398 | 2026-03-01 |
| `ecc-tools/oelala-1787504296949` | +12 / −112 | 2026-08-23 |
| `copilot/unified-generation-core-adapter-architecture` | +15 / −191 | 2026-04-16 |
| `feat/compute-backend-inventory` | +0 / −78 | 2026-08-23 |
| `feat/compute-node-types` | +0 / −74 | 2026-08-24 |
| `feat/minimax-h3-lora` | +0 / −69 | 2026-08-24 |
| `feat/minimax-h3-lora-cloud` | +0 / −57 | 2026-09-23 |

Per branch, oldest first, decide with evidence and report the decision:

- **still wanted** → rebase onto `origin/main`, verify, open one PR (`Closes #<issue>`,
  creating the issue first when none exists);
- **already superseded** (its change is on `main` in another form, or the model/feature
  was retired — e.g. Wan 2.2 retired 2026-10-01) → report it as superseded with the
  commit or doc that proves it, and recommend deletion to the operator. Do not delete.
- **conflicting or unclear** → fail closed: report with the diff summary.

`+0` branches are already contained in `origin/main` — verify with
`git log origin/main --oneline --grep`/`git cherry` before calling them dead.

## 6. Phase 3 — Advance the roadmap (only if budget remains)

Pick the next step from `docs/ROADMAP.md` (and `docs/TODO_LIST.md` for concrete
leftovers), write a falsifiable issue (problem statement, acceptance criteria,
evidence links), then implement it in the same run — **one step maximum**.

## 7. Phase 4 — Report

Post one comment on the tracking issue **#210 "Oelala loop — nightly run log (04:00 UTC)"**
in `m0nklabs/oelala` (`gh issue comment 210 --body-file <tmp>`):

```markdown
## Run <YYYY-MM-DD> 04:00 UTC
- window: <start> → <end> UTC
- PRs: #<n> <one line: what changed / settled / blocked> (or "none open")
- issues: #<n> <state after the run> (or "no executable issue")
- pipeline: <branch decisions from §5, or "not needed">
- evidence: <commit SHAs, PR links, test counts, CI state>
- next intended action: <one line>
- blockers / operator decisions needed: <one line or "none">
- restarts needed: <what must be restarted and why — never performed here>
```

Report facts, not intentions: failed or unrun checks are named as such. Also send the
MoniFuse agent-activity events (`task_started`, then `task_achieved` / `task_failed` /
`blocked`) per the `agent-activity-log` skill (fail-open).

## 8. Repo conventions that bite

- **Changelog**: every PR adds `changelog/{NNN}-{desc}.md` with `### Added/Fixed/Changed`.
  Never edit root `CHANGELOG.md` — `scripts/merge_changelog.py` merges on release.
- **Artifacts in English**; chat with the operator in Dutch.
- **Frontend** is JSX-only; use `apiFetch()` from `utils/api.ts`, never raw `fetch()`.
- **Media dirs**: only canonical `media/generated/`, `ComfyUI/output/`, `uploads/`.
- **Logs are append-only**: never rewrite, trim or delete old log entries.
- **Docs are not deleted or pruned**; keep `docs/GENERATION_MODES_TREE.md` current
  after successful generation runs; LoRA metadata lives in `docs/lora_registry.yaml`.
- **No hardcoded addresses** in committed code, tests or docs — use env/config.
- **Debug logging** in this repo uses emoji prefixes (🐛🔍⚠️❌✅) with timestamps.

## 9. Failure handling

- Uncertain finding or unclear intent → fail closed: leave it for the human with evidence.
- Red CI caused by the loop's own push → fix it in the same run when in scope; otherwise
  report it. Never "fix" a gate by weakening tests.
- Red CI or failing tests that predate the run → report; do not expand scope.
- GitHub, the runner pool or the worktree unavailable → report blocked, make no changes.
- A run that produces nothing is a valid outcome: say so plainly.

## 10. Scheduling (one-time setup, done by the operator's session)

The reminder is a durable DSH Schedule task bound to the loop Session, created with
the `schedule_create` tool (mounted by the `standard`, `cordis` and `ptc` presets —
not by the Agent-Teams lead preset):

```json
{
  "title": "Oelala loop — nightly development run",
  "daily": { "time": "04:00:00", "time_zone": "UTC" },
  "prompt": "Oelala loop run. Read <repo>/docs/OELALA_LOOP.md and execute it end-to-end for today, exactly as written: preflight in the loop worktree, open-PR remediation incl. PR-Piet suggestions and /review requests, open issues, at most one implementation (or one pipeline-seeding action when nothing is open), then post the run report as a comment on the tracking issue. Hard stops: never merge a PR, never push to main, never force-push, never approve, never restart any systemd unit, never touch the operator clone or another repository."
}
```

Verify with `schedule_list`, then prove delivery once with a short `after_seconds`
test reminder before trusting the 04:00 run. The Host restores a cold Session when
delivery is due, so the loop survives restarts and missed nights.
