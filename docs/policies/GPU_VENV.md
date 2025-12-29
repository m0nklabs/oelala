# Canonical GPU Virtual Environment

This server uses a single canonical Python virtual environment for GPU/ML work.

## Canonical location

- Canonical venv path: `/home/flip/venvs/gpu`
- Implementation detail: this is a symlink that currently points to `/home/flip/venvs/torch-sm120`.

## Why

- Avoids duplicated multi-GB environments across projects.
- Keeps CUDA/Torch/Diffusers tooling consistent.
- Makes it easier to archive legacy environments safely.

## Usage

- Activate: `source /home/flip/venvs/gpu/bin/activate`
- Run Python explicitly: `/home/flip/venvs/gpu/bin/python your_script.py`

## Archiving legacy venvs

When a GPU-related venv is no longer the default:

1. Move it to `/home/flip/venvs/_archive/YYYY-MM-DD/<name>`
2. Replace the original path with a symlink to the archived location (or to the canonical venv when compatible).

This preserves existing hard-coded paths while keeping the canonical venv as the default.
