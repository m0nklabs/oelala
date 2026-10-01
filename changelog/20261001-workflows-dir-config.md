### Changed
- The workflow directory is no longer hardcoded to one machine's layout.
  `comfyui_client.WORKFLOWS_DIR` was pinned to an absolute
  `/home/flip/oelala/workflows` path, which is environment-specific state in
  committed code and would break on any other host or checkout. It now defaults
  to the repository's `workflows/` directory resolved relative to the module
  (`Path(__file__).resolve().parents[2] / "workflows"`) and can be overridden with
  `OELALA_WORKFLOWS_DIR` for deployments that keep workflows elsewhere. The
  default resolves to the same directory as before, so behaviour is unchanged
  when the variable is unset; verified that `load_workflow_from_file()` still
  loads the MiniMax-H3 T2V workflow and that the override takes effect.
- `tests/gpu/test_integration.py` no longer pins the workflows path either. The
  fixture returned the same absolute literal, so the test silently skipped on any
  other host; it now derives the repository root from `__file__` like the model
  directory helper next to it.
