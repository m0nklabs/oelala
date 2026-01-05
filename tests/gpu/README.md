# GPU Integration Tests

Tests in this directory run on the self-hosted GPU runner and have access to:
- Local ComfyUI (localhost:8188)
- Local Backend API (localhost:7998)
- GPU with VRAM
- All installed models

## Running locally

```bash
source /home/flip/venvs/gpu/bin/activate
cd /home/flip/oelala
pytest tests/gpu/ -v
```

## Running via GitHub Actions

Trigger manually or push changes to `src/backend/comfyui_client.py` or `workflows/`.

## Test markers

- `@pytest.mark.slow` - Tests that do actual GPU generation (skipped by default)

Run slow tests:
```bash
pytest tests/gpu/ -v -m slow
```
