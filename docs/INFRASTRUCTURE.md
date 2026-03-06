# Oelala Infrastructure Inventory

> Last Updated: 2026-03-06

This document is the canonical infrastructure snapshot for Oelala.

## Repositories

| Repository | Purpose |
|------------|---------|
| `m0nklabs/oelala` | main product: frontend, backend, workflows, docs, gallery, billing, admin |
| `m0nklabs/oelala-storage` | object storage, retention, signed URLs, node sync/replication groundwork |
| `m0nk111/llama-cpp-guardian` | local LLM proxy/control plane for prompt and analysis support paths |

## Capability Ownership

| Capability | Primary Repo |
|------------|--------------|
| Product UI, tool workflows, gallery, credits, admin | `oelala` |
| Object storage, metadata, dedup, GC, signed access | `oelala-storage` |
| Local LLM proxying, model switching, benchmarks, sessions | `llama_cpp_guardian` |

## Core Services

| Service | Machine | Port | systemd unit |
|---------|---------|------|--------------|
| frontend | ai-kvm2 | 5174 | `oelala-frontend.service` |
| backend | ai-kvm2 | 7998 | `oelala-backend.service` |
| ComfyUI | ai-kvm2 | 8188 | `comfyui.service` |
| storage primary | ai-kvm2 | 7990 | `oelala-storage.service` |
| storage node-01 | ai-kvm2 | 7993 | `oelala-node-01.service` |

## Machines

| Machine | Address | Role |
|---------|---------|------|
| `ai-kvm2` | `192.168.1.35` | main Oelala host |
| `ubuntu-oelalastorage2` | `192.168.1.62` | remote storage node |

## Public Hostnames

| Hostname | Route |
|----------|-------|
| `oelala.xyz` | frontend |
| `api.oelala.xyz` | backend |
| `storage.oelala.xyz` | storage primary public endpoint |
| `storage2.oelala.xyz` | remote storage node public endpoint |
| `storage-main.oelala.xyz` | explicit primary-node naming used in storage rollout docs/config |
| `storage-node-01.oelala.xyz` | explicit local node-01 naming used in storage rollout docs/config |

## Cloudflare Tunnels

| Tunnel | ID | Machine |
|--------|----|---------|
| `oelala-main` | `b34ce27b-e9b1-4926-b5fe-ebbaf42d506a` | ai-kvm2 |
| `oelala-storage-node2` | `83d253c4-24eb-4643-b36f-174a2fc3f10b` | ubuntu-oelalastorage2 |

Rule: each node should own its own tunnel rather than depending on another node for ingress.

## Storage Nodes

| Node | Role | Notes |
|------|------|-------|
| primary | coordinator / primary | main storage entrypoint |
| node-01 | additional local node | separate local service ports |
| node-02 | remote node | autonomous remote host |

## GPU Inventory

| GPU | VRAM | CUDA |
|-----|------|------|
| RTX 3060 | 12GB | cuda:0 |
| RTX 5060 Ti | 16GB | cuda:1 |

Preferred DisTorch2 allocation:

```text
cuda:0,10gb;cuda:1,15gb;cpu,*
```

## Storage/Auth Conventions

### Storage API

```http
Authorization: Bearer <token>
```

### Storage Admin

```http
X-Admin-Secret: <secret>
```

### Retention

```http
X-Expires-At: <RFC3339 timestamp>
```

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