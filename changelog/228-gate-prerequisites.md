### Fixed

- The CI job that pushed style fixes to pull-request branches is gone. It committed with
  `GITHUB_TOKEN`, and GitHub does not start workflows for such pushes — so the pushed head
  carried no CI at all (`798…`: 18 checks with the backend gate, the bot's follow-up heads:
  2). With required status checks that makes a PR permanently unmergeable, which is exactly
  what a merge gate must not do. `Pre-commit checks` already gates style, so the job was a
  convenience that broke the gate.
- `.github/CODEOWNERS` no longer carries another project's boilerplate ("CODEOWNERS for
  cryptotrader", `/core/execution/`, `/frontend/`). It now scopes exactly the irreversible
  five — auth, migrations, secrets, deployment, dependency manifests — plus
  `.github/workflows/`, and deliberately has no catch-all so the autonomous loop can still
  merge anything else once it has merge rights.
