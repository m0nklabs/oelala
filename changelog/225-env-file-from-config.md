### Fixed

- The backend no longer hardcodes the env file it loads. `app.py` called
  `load_dotenv(dotenv_path="/home/<user>/oelala/.env")`, an absolute host path in committed
  code (AGENTS.md rule 6): every process started from another clone or worktree silently
  loaded the first deployment's production configuration, and a checkout anywhere else had
  no configuration at all. The path now comes from `OELALA_ENV_FILE`, or is derived from the
  module location (`<checkout>/.env`), so each copy reads its own file.
- An `OELALA_ENV_FILE` that points at a missing file is reported instead of being ignored
  silently; an absent default is allowed.

### Changed

- `.env.example` documents `OELALA_ENV_FILE`.
