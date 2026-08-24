# Oelala Infrastructure Inventory

> Last Updated: 2026-07-14

This document is the canonical infrastructure snapshot for Oelala.

## Repositories

| Repository | Purpose |
|------------|---------|
| `m0nklabs/oelala` | main product: frontend, backend, workflows, docs, gallery, billing, admin |
| `m0nklabs/oelala-storage` | ⚠️ **Deprecated** — replaced by MinIO. Historical Go-based storage service. |
| `m0nk111/llama-cpp-guardian` | local LLM proxy/control plane for prompt and analysis support paths |

## Capability Ownership

| Capability | Primary Repo |
|------------|--------------|
| Product UI, tool workflows, gallery, credits, admin | `oelala` |
| S3-compatible object storage (MinIO) | MinIO (local service) |
| Local LLM proxying, model switching, benchmarks, sessions | `llama_cpp_guardian` |

## Core Services

| Service | Machine | Port | systemd unit |
|---------|---------|------|--------------|
| frontend | ai-kvm2 | 5174 | `oelala-frontend.service` |
| backend | ai-kvm2 | 7998 | `oelala-backend.service` |
| ComfyUI | ai-kvm2 | 8188 | `comfyui.service` |
| MinIO S3 API | ai-kvm2 | 9000 | `minio.service` |
| MinIO Console | ai-kvm2 | 9001 | `minio.service` |

## Machines

| Machine | Address | Role |
|---------|---------|------|
| `ai-kvm2` | `LAN` | main Oelala host |
| `ubuntu-oelalastorage2` | `LAN` | remote storage node |

## Public Hostnames

| Hostname | Route |
|----------|-------|
| `oelala.xyz` | frontend |
| `api.oelala.xyz` | backend |
| `storage.oelala.xyz` | MinIO S3 API (via Cloudflare tunnel) |

## Cloudflare Tunnels

| Tunnel | ID | Machine |
|--------|----|---------|
| `oelala-main` | `b34ce27b-e9b1-4926-b5fe-ebbaf42d506a` | ai-kvm2 |
| `oelala-storage-node2` | `83d253c4-24eb-4643-b36f-174a2fc3f10b` | ubuntu-oelalastorage2 |

Rule: each node should own its own tunnel rather than depending on another node for ingress.

## Storage

Storage is provided by **MinIO** (S3-compatible object storage), replacing the previous custom oelala-storage Go service.

| Component | Details |
|-----------|---------|
| Service | MinIO |
| S3 API port | 9000 |
| Console port | 9001 |
| systemd unit | `minio.service` |
| Health check | `/minio/health/live` |
| Access | MinIO access key / secret key |

## GPU Inventory

| GPU | VRAM | CUDA |
|-----|------|------|
| RTX 3060 | 12GB | cuda:0 |
| RTX 5060 Ti | 16GB | cuda:1 |

Preferred DisTorch2 allocation:

```text
cuda:0,10gb;cuda:1,15gb;cpu,*
```

## Storage Access

### MinIO S3 API

Authentication uses MinIO access key / secret key (configured in `.env`):

```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=<access-key>
MINIO_SECRET_KEY=<secret-key>
```

### Presigned URLs

The backend generates S3 presigned URLs for time-limited media access.

### Retention

Retention is managed via MinIO bucket lifecycle rules configured by the backend.

## External Vendors / Platforms

| Vendor | Usage |
|--------|-------|
| Cloudflare | tunnels, DNS, proxy/cache |
| Supabase | auth, database, app state |
| Stripe | credits/payments |
| RunPod | cloud GPU execution |
| GitHub / GHCR | source control, CI, worker images |
| Hugging Face | model distribution |

## Canonical Rule

If a hostname, service port, tunnel, or auth header changes, update this file and then sync the more specific docs that reference it.
