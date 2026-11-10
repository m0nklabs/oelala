# Oelala Loop — development run procedure (cold file)

> Executed by a cron reminder bound to the **oelala-loop Session**: **4× per day at
> 04:00, 10:00, 16:00 and 22:00 UTC**. This file is the authoritative procedure: the
> reminder only says "run the oelala loop", so a compacted or restored Session re-reads
> this document and continues here.
>
> Operator decisions baked in: autonomy **everything except merge** and scope **this
> repository only** (2026-10-25); cadence **4× per day** (2026-10-27, replacing the
> single 04:00 UTC nightly run — the guardian loop still runs once at 04:00 UTC).

## 0. Environment (verified 2026-10-25)

| Fact | Value |
|---|---|
| Repository | `m0nklabs/oelala` (public; root `README.md` is a deliberate privacy decoy — never "fix" it) |
| Tracking issue | **#210** — the run log (one report per run, heading carries the run time) |
| Loop worktree | `$HOME/oelala-loop` — branch `loop/workspace`, tracks `origin/main` |
| Operator clone | `$HOME/oelala` — **hands off**: it carries the operator's uncommitted work |
| Truth for code | `origin/main` (fetch first; the operator clone is often behind) |
| Test tooling | `ruff` 0.15.15 on PATH. The system `python3` **cannot** import `app` (no `sentry_sdk`); for a real test run use the read-only GPU venv `$HOME/venvs/gpu/bin/python -m pytest`, or a throwaway venv from `requirements-web.txt`. No project `.venv` — do not create a heavy one |
| GPU venv | `$HOME/venvs/gpu` is production-critical (`chattr +i`) — read-only use only, never delete/move/recreate |

Work only inside the loop worktree:

```bash
export WT="${OELALA_LOOP_WT:-$HOME/oelala-loop}"
cd "$WT"
git status --porcelain          # MUST be empty; anything here = report blocked and stop
git fetch origin --prune
git switch loop/workspace && git pull --ff-only
```

The cleanliness check comes **first**. `git switch` and `git pull` both succeed with local
edits present as long as they do not overlap the incoming changes, so a dirty worktree
would otherwise be modified before §3 gets a chance to stop the run.

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
- comment the run report on the tracking issue.

**Not granted (hard stops):**

1. **Merging is allowed only under §4.2** (24 hours after the review is finished, with the
   latest activity read and the merge re-verified at merge time). Never approve a PR, never dismiss a
   review, never resolve a thread that still has an unanswered finding. Everything else is
   left ready for the human merge.
2. **Never push to `main`** or any long-lived branch, never force-push, never rewrite
   published history, never delete branches or issues.
3. **Never restart or stop a systemd unit** (`oelala-backend`, `oelala-frontend`,
   `comfyui`, `minio`, and `frigate` especially). Validation only; a restart is a
   recommendation in the report.
4. **Never touch the operator clone** `$HOME/oelala` (dirty tree = their work), never
   stash/checkout/reset/clean it.
5. **Never touch `$HOME/venvs`** (production runtime, `chattr +i`) and never run
   `npm run dev`, `uvicorn` or `python main.py` — services are systemd-only.
6. **Frigate is hands off** (security cameras, 24/7): read-only config, never "fix" it.
7. **Never touch another repository.** Cross-project findings are reported, not fixed.
8. **Never commit secrets**; never add private IPs, hostnames or environment-specific
   values to committed files (read them from `.env` / `compute_backends.json`).
9. **Never use `git stash` in the loop worktree.** It shares `.git` with the operator
   clone (`git rev-parse --git-common-dir` → `$HOME/oelala/.git`), so `refs/stash` is one
   stack for both checkouts: a `stash pop` here can pull the operator's WIP into this
   tree and a `stash drop` can destroy it. Park work on a throwaway branch instead.
   (2026-10-27: exactly that happened. A dropped stash becomes unreachable and is only
   recoverable until Git garbage-collects it — `gc.pruneExpire` defaults to two weeks,
   but `gc --prune=now` is immediate. Recover promptly with
   `git stash store -m "<message>" <sha>` using the sha the drop printed; after pruning
   the work is gone, so recovery is a race, not a guarantee.)

## 2. Budget per run

Four runs a day multiply everything, so the budget is **per run** and deliberately small:

- **all** open-PR remediation (batched: one coherent tested push per PR head);
- **at most one** new implementation (one issue or one roadmap step), from a branch
  off `origin/main`;
- **at most one** pipeline-seeding action (§5) when PRs and issues are empty;
- **no new PR while 3 or more loop PRs are already open awaiting human merge** — the
  merge queue, not the agent, is the bottleneck. In that state, deepen what is open
  (findings, verification, evidence) and report the queue instead of opening more;
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
gh pr view <N> -R m0nklabs/oelala --json reviews,comments,files,commits,statusCheckRollup,reviewDecision,headRefName,headRefOid
gh api --paginate "repos/m0nklabs/oelala/pulls/<N>/comments?per_page=100"  # ALL inline threads
gh pr checks <N> -R m0nklabs/oelala
```

Work on the **PR head**, never on `loop/workspace`: check it out explicitly and verify the
SHA before touching anything, then push back to that exact head branch (a bare `git push`
depends on the local upstream config and can miss the PR entirely):

```bash
git fetch origin --prune
BR="review/<N>-<headRefName>"
# A previous run may have committed but failed to push. The worktree is clean again by
# then, so `switch -C` would silently point the branch away from those commits:
# refuse instead of destroying an earlier run's work.
if git rev-parse --verify --quiet "$BR" >/dev/null &&
   ! git merge-base --is-ancestor "$BR" "origin/<headRefName>"; then
    echo "BLOCKED: $BR holds commits that are not in origin/<headRefName>"; exit 1
fi
git switch -C "$BR" "origin/<headRefName>"
test "$(git rev-parse HEAD)" = "<headRefOid>" || { echo "head mismatch"; exit 1; }
# … fix, test, commit …
git push origin HEAD:<headRefName>
```

### 4.1 Check what changed since the last run — before fixing anything

An open PR is not static: the base branch moves, reviewers comment, bots push, and a merged
PR can invalidate the premise of an open one. Start every run by diffing the world since the
last run-log entry, then respond to *that*:

```bash
# Activity on the PR itself. Read every page and do not slice the result: REST returns these
# ascending, so any .[-N:] both hides the newest entries and truncates the delta. Read the
# entries newer than the last run-log comment instead.
gh api --paginate --slurp "repos/m0nklabs/oelala/issues/<N>/comments?per_page=100" \
    --jq 'add | .[] | "\(.created_at) \(.user.login)"'
gh api --paginate --slurp "repos/m0nklabs/oelala/pulls/<N>/reviews?per_page=100" \
    --jq 'add | .[] | "\(.submitted_at) \(.state)"'
gh pr checks <N> -R m0nklabs/oelala --json name,state
# how far the base moved, and whether the BASE side touched the PR's files. Diff from the
# merge-base, not head..origin/main: a two-dot diff includes the PR's own files and would
# always conclude "the base touched them".
git rev-list --count "$(git merge-base <headRefOid> origin/main)"..origin/main
git diff --name-only "$(git merge-base <headRefOid> origin/main)" origin/main | head -20
```

Rules for what you find:

- **Base moved and touches the PR's files** → merge `origin/main` into the PR head (never
  force-push), re-run the suite against the current pins, and say so in a comment. Green
  checks from before a dependency merge prove nothing about the merged state.
- **New review or comments** → answer them; findings get fixed or refuted with evidence.
- **Something that changes the PR's premise** (a merged PR introduced a helper this PR
  duplicates, a config knob that replaced a literal) → treat it as a finding: adapt the code
  and pin the relationship with a test, so the two cannot drift apart later.
- **Nothing new** → say so in one line with the timestamp and move on. Do not pad a report.

Requested by the operator on 2026-11-03 (translated from Dutch): *"You are also supposed to
check open items for new additions, and then respond to those."*

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
# pytest.ini registers the `gpu`/`integration` markers but does not exclude them, and
# tests/gpu calls the live backend and ComfyUI (POSTs included). Exclude both here;
# GPU/integration runs belong to the dedicated runner.
$HOME/venvs/gpu/bin/python -m pytest tests/ -x -q \
    --ignore=tests/gpu --ignore=tests/manual -m "not gpu and not integration"
ruff check src/                          # plus ruff format --check on touched files
git commit … && git push origin HEAD:<headRefName>   # branch only
gh pr comment <N> -R m0nklabs/oelala --body '/review'
```

When a PR is settled (CI green, latest review covers the merge head, every finding
resolved or answered), leave one closing comment: what changed, the evidence, and
`Ready for human merge`. Merging follows §4.2 only. Draft PRs stay draft until validated.

### 4.2 Merge readiness and the 24-hour rule

Merge-readiness means the review is **finished**, not merely quiet: every outstanding review is
closed out — with a fix, or with a reply that explains with evidence why the change is not needed.
When there was nothing to review, the PR is merge-ready as soon as it is up and its checks pass.
A closing comment that says what the change does is the normal way to mark that point, but it is
**never a precondition**. This mirrors the user-level rule in `$DSH_HOME/AGENTS.md`.

Merge **24 hours after** that point if the operator has not intervened. Operator silence for 24
hours is standing permission — do not ask again at merge time. A quiet PR is not a stalled one: no
comment on it and 24 hours gone means merge it.

**The clock.** It starts when the last outstanding review is answered, or — with nothing to review
— at the last push on the head. Not at the PR's creation. An unanswered review never becomes
eligible through age, and answering a newer review does not absolve an older one. Restart the
clock whenever new work is pushed.

**Read the latest activity before every merge.** A reply from the operator, a reviewer or a bot can
carry a question, a suggestion, a correction or context. Fetch that PR's latest comments, reviews
and commits and read them — never scan for a veto signature only. If the last activity still needs
a reply, a fix or a decision, do that instead of merging, and the clock restarts. A recent
`updated_at`, a green check or yesterday's state proves nothing.

Then re-verify at merge time — an old green check says nothing about `main` today. The
verification does **not** push anything: a pushed refresh is new work, and new work restarts the
24-hour clock, so a refresh-then-merge in one run would be impossible by construction.

```bash
git fetch origin --prune
gh pr view <N> -R m0nklabs/oelala --json mergeStateStatus,mergeable,reviewDecision
gh api repos/m0nklabs/oelala/commits/<headSHA>/check-runs \
    --jq '[.check_runs[]|select(.name|test("Backend|Pre-commit|Frontend"))]|map(.conclusion)'

# 1. Would it still merge cleanly into today's main? Ask git, not a pipe:
#    `merge-tree --write-tree` writes a tree *and* exits 1 on a conflict, so
#    piping it through `head` reports success for exactly the case that matters.
if git merge-tree --write-tree origin/main origin/<headRefName> >/dev/null; then
    echo "merge is clean"
else
    echo "CONFLICT — refresh the base; that push is new work, so the clock restarts"
fi

# 2. Test the RESULT of the merge, not the branch. A conflict-free merge can still
#    break when main changed an interface this branch still calls. A throwaway
#    worktree gives the combined tree without touching the branch or pushing.
WT=$(mktemp -d)
# Clean up on every exit path, so an abort never leaves a worktree behind.
trap 'git worktree remove --force "$WT" 2>/dev/null' EXIT
git worktree add -q "$WT" origin/<headRefName>
# Stop here on a failed merge or a failed test: the cleanup must not turn a red
# suite into a green-looking step, and the merge command below must not run.
( cd "$WT" && git merge -q --no-edit origin/main && \
  $HOME/venvs/gpu/bin/python -m pytest tests/ -x -q \
      --ignore=tests/gpu --ignore=tests/manual -m "not gpu and not integration" ) || {
    echo "❌ the merged tree does not pass — do NOT merge"; exit 1; }
trap - EXIT; git worktree remove --force "$WT"

# --match-head-commit binds the merge to the commit that was just verified: without it a
# commit pushed between verification and merge would be merged on the strength of checks
# and a review that never saw it.
gh pr merge <N> -R m0nklabs/oelala --merge --delete-branch=false \
    --match-head-commit "<headSHA>"
```

CI already tests GitHub's synthetic merge ref for a `pull_request` event — but only for the merge
state **at the time it ran**. If `main` moves afterwards, that combination is untested and, with
`strict_required_status_checks_policy: false`, still mergeable. Step 2 is what covers that gap.

Refresh the base with a merge commit **only when the dry run conflicts**, and treat that push as
new work: the clock restarts and the next run merges. `strict_required_status_checks_policy` is
`false` on this repo, so a green head does not need a base update to be mergeable.

`gh pr merge` answers a refused merge with "the base branch policy prohibits the merge" and
offers `--admin`. **Never use `--admin`** (or any other bypass): the policy that just refused the
merge is the gate, and a PR it refuses is a finding to report, not a rule to override. The same
goes for `--auto` — it is not a bypass, but it hides the verification this section requires, so
the loop merges explicitly or not at all.

Stop and report instead of merging when: a required check is not green on the head, the review
on that head is unanswered or requests changes, the base merge conflicts, `main` itself is red,
or the PR touches the human-gated five (auth, migrations, secrets, deploy, dependency majors) —
those keep their gate regardless of the clock. At most **three** merges per run.

**A merge is a deliberate action on one PR**, never a side effect of a watch, a sweep or a batch
scan: read that PR's activity, re-verify it, and merge only that one. Record the anchor in the
merge comment — `answered-review` with the link, or `auto-merged after N hours` (or `N days`) with
the re-verification result — so the timing stays reconstructable.

"Settled" is a checklist, not a feeling. A PR is **not** finished until 1–4 hold:

1. CI green on the current head SHA.
2. The latest PR-Piet review is on that **same** SHA — a review of an older head proves
   nothing about the current one.
3. Every finding is resolved or answered, and no thread is unanswered.
4. Any finding the review marked `[backlog]` is recorded somewhere durable (an issue), not
   just mentioned in passing.
5. **The point is recorded**, so the clock stays reconstructable — a closing comment naming that
   SHA when one is written, or the merge comment when it is not. Recording is after the fact:
   **it is never a precondition for merging, with or without a completed review.**

Points 1–4 are the gate; (5) is bookkeeping. A PR whose reviews are all answered is merge-ready the
moment it is green, comment or no comment, and its clock is running from the last answered review.
When there was a review, answer it — an unanswered review blocks regardless of age and never
becomes eligible by waiting. If a run cannot settle a PR, the run report says so explicitly instead
of leaving it implicit. (Learned the hard way on 2026-11-03: #225 was green and review-covered and
the run moved on without closing it out, so the human had to ask. Revised 2026-11-09: the operator
made the record explicit but non-blocking — 24 hours, silence is permission, and a quiet PR is not
a stalled one.)

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

- **still wanted** → rebase onto `origin/main` and push the result to a **new** branch
  (e.g. `revive/<slug>`) — never force-push the existing remote branch, see §1.2 — then
  verify and open one PR (`Closes #<issue>`, creating the issue first when none exists);
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

Post one comment on the tracking issue **#210** in `m0nklabs/oelala`
(`gh issue comment 210 --body-file <tmp>`). With four runs a day the heading must carry
the actual run time, otherwise the log becomes unreadable:

```markdown
## Run <YYYY-MM-DD HH:MM> UTC
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
  "title": "Oelala loop — development run",
  "cron": { "expression": "0 4,10,16,22 * * *", "time_zone": "UTC" },
  "prompt": "Oelala loop run. Read <repo>/docs/OELALA_LOOP.md and execute it end-to-end for THIS run, exactly as written: preflight in the loop worktree, open-PR remediation incl. PR-Piet suggestions and /review requests, open issues, at most one implementation (or one pipeline-seeding action when nothing is open), then post a run report as a comment on the tracking issue whose heading carries this run's UTC date and time. Hard stops: never merge a PR, never push to main, never force-push, never approve, never restart any systemd unit, never touch the operator clone or another repository."
}
```

Verify with `schedule_list`, then prove delivery once with a short `after_seconds`
test reminder before trusting the schedule. The Host restores a cold Session when
delivery is due, so the loop survives restarts and missed runs.

### Created state (verified 2026-10-25, DSH home `$HOME/.dsh-next`)

| Item | Value |
|---|---|
| Task id | `<schedule-task-id>` — resolve with `schedule_list` |
| Cadence | cron `0 4,10,16,22 * * *`, zone `UTC` (= 05:00/11:00/17:00/23:00 CET in winter, one hour later in CEST) |
| Next run | `<next-run-iso8601>` — resolve with `schedule_list` |
| State / mode | `active`, `deliveryMode: host` |
| Created from | a `standard`-preset session; `schedule_create` is mounted there |

Delivery was proven, not assumed: a one-shot `after_seconds: 120` test reminder
(`<test-task-id>`) fired within milliseconds of its due time and was deleted afterwards.

Mechanics behind the guarantee: the `dsh-web-next.service` unit for this DSH home runs
persistently and owns the scheduler; `@deepseek-ai/dsh-schedule` resolves the bound
session with `sessionController.resolveAgent(task.sessionId)` when a task is due, and a
recurring task that missed several slots delivers only its latest missed occurrence — it
does not catch up on all four. If that unit is down at a slot the run is missed for good:
check `systemctl --user is-active dsh-web-next` when runs look silent.

The guardian loop keeps the earlier convention: one daily `04:00:00` UTC task (id via
`schedule_list`, title `Guardian loop — nightly development run`), procedure in
`$HOME/guardian-llmprovider-gateway/docs/GUARDIAN_LOOP.md`.

The reminder prompt resolves this file in order, and **only** from permitted sources:

1. `$WT/docs/OELALA_LOOP.md` — the loop worktree, which tracks `origin/main` once this
   branch is merged;
2. `git show origin/main:docs/OELALA_LOOP.md` — the merged copy on `origin`;
3. `git show origin/docs/oelala-loop-procedure:docs/OELALA_LOOP.md` — this branch on
   `origin` while the PR is still open.

The operator clone `$HOME/oelala` is **never** consulted (§1.4), not even read-only. If
none of the three resolve, report blocked (§9) instead of looking elsewhere.
