# Self-Hosted GitHub Actions Runners

This server hosts self-hosted GitHub Actions runners for oelala projects.

## Runners

### oelala-gpu (for m0nklabs/oelala)
- **Location**: `/home/flip/actions-runner/`
- **Labels**: `self-hosted`, `Linux`, `X64`, `gpu`, `comfyui`, `ci`
- **Service**: `actions.runner.m0nklabs-oelala.oelala-gpu.service`

### oelala-storage-runner (for m0nklabs/oelala-storage) — DEPRECATED
> **Note**: oelala-storage has been replaced by MinIO. This runner may be decommissioned.
- **Location**: `/home/flip/actions-runner-storage/`
- **Labels**: `self-hosted`, `Linux`, `X64`, `ci`, `go`
- **Service**: `actions.runner.m0nklabs-oelala-storage.oelala-storage-runner.service`

## Installed Tools

| Tool | Version | Path |
|------|---------|------|
| Go | 1.22.4 | `/usr/local/go/bin/go` |
| golangci-lint | 2.8.0 | `/usr/local/bin/golangci-lint` |
| Python | 3.12.3 | `/usr/bin/python3` |
| Node.js | 22.21.0 | `/usr/bin/node` |
| npm | 10.9.4 | `/usr/bin/npm` |
| Docker | 28.2.2 | `/usr/bin/docker` |
| ruff | latest | `/home/flip/.local/bin/ruff` |
| pre-commit | latest | `/home/flip/.local/bin/pre-commit` |

## Management Commands

```bash
# Check runner status
sudo systemctl status actions.runner.m0nklabs-oelala.oelala-gpu.service
sudo systemctl status actions.runner.m0nklabs-oelala-storage.oelala-storage-runner.service

# View logs
journalctl -u actions.runner.m0nklabs-oelala.oelala-gpu.service -f
journalctl -u actions.runner.m0nklabs-oelala-storage.oelala-storage-runner.service -f

# Restart runners
sudo systemctl restart actions.runner.m0nklabs-oelala.oelala-gpu.service
sudo systemctl restart actions.runner.m0nklabs-oelala-storage.oelala-storage-runner.service
```

## Workflow Configuration

All workflows should use:
```yaml
runs-on: [self-hosted, Linux]
```

For GPU-specific jobs:
```yaml
runs-on: [self-hosted, Linux, gpu]
```

For Go projects:
```yaml
runs-on: [self-hosted, Linux, ci, go]
```

## Benefits

1. **No download time** - All tools pre-installed
2. **Persistent cache** - Go modules, npm packages cached
3. **GPU access** - RTX 5060 Ti + RTX 3060 available
4. **No minutes limit** - No GitHub Actions billing
5. **Faster execution** - Local NVMe storage, dedicated CPU

## Updating Tools

```bash
# Update golangci-lint
cd /tmp && curl -sSfL -o golangci-lint.tar.gz \
  https://github.com/golangci/golangci-lint/releases/download/vX.Y.Z/golangci-lint-X.Y.Z-linux-amd64.tar.gz \
  && tar xzf golangci-lint.tar.gz \
  && sudo mv golangci-lint-X.Y.Z-linux-amd64/golangci-lint /usr/local/bin/

# Update Python tools
pip install --upgrade ruff pre-commit

# Update Node (via nvm or apt)
sudo apt update && sudo apt install nodejs
```
