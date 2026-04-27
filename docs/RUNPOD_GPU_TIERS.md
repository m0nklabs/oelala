# RunPod Serverless GPU Tier IDs

> **Last updated**: 2026-03-07
> **Source**: Discovered by enabling all GPU checkboxes in RunPod UI and reading back via API.

## Critical: GPU ID Format

RunPod's API `gpuIds` field expects **architecture-tier IDs**, NOT GPU model names.

**The API silently accepts ANY string** — it will happily store `"NVIDIA GeForce RTX 4090"` without error, but the **scheduler will never match it**. This cost us hours of debugging (zero workers provisioned with wrong format).

### Format Pattern

```
{ARCHITECTURE}_{VRAM_GB}          → Consumer / community GPUs
{ARCHITECTURE}_{VRAM_GB}_PRO      → Professional / datacenter GPUs
```

## Complete Tier Reference (11 tiers as of 2026-03-07)

| Tier ID | Architecture | VRAM | Typical GPUs | Notes |
|---------|-------------|------|--------------|-------|
| `AMPERE_16` | Ampere | 16 GB | RTX A4000 | Low supply |
| `AMPERE_24` | Ampere | 24 GB | RTX 3090, RTX 3090 Ti | Consumer Ampere |
| `AMPERE_48` | Ampere | 48 GB | A40, RTX A6000 | Datacenter Ampere |
| `AMPERE_80` | Ampere | 80 GB | A100 80GB (SXM/PCIe) | High-end datacenter |
| `ADA_24` | Ada Lovelace | 24 GB | RTX 4090, L4 | Consumer Ada |
| `ADA_32_PRO` | Ada Lovelace | 32 GB | RTX 5000 Ada | Professional Ada |
| `ADA_48_PRO` | Ada Lovelace | 48 GB | L40S, RTX 6000 Ada | Professional Ada |
| `ADA_80_PRO` | Ada Lovelace | 80 GB | (Multi-GPU / specialized) | Rare |
| `HOPPER_141` | Hopper | 141 GB | H200 | HBM3e |
| `BLACKWELL_96` | Blackwell | 96 GB | B200 | Next-gen |
| `BLACKWELL_180` | Blackwell | 180 GB | GB200 / B300 | Low supply |

## oelala-cloud-wan22 Endpoint Config

**Endpoint ID**: `x2x496ymkidl3m`

**Template**: `tkpy0pi8gt` with `containerDiskInGb=100`

Currently configured with **48GB+ tiers only** on the active endpoint (for 20 sec / 321 frame WAN 2.2 video @ ~26GB VRAM):

```
AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO,BLACKWELL_96,HOPPER_141,BLACKWELL_180
```

### Current Production Defaults

- `workersMin=0` keeps burst traffic cost-effective while allowing scale-to-zero.
- `workersMax=2` allows one active job plus one burst/cold-start slot without runaway spend.
- `idleTimeout=120` keeps a warm worker available briefly after a job.
- `scalerType=QUEUE_DELAY`, `scalerValue=4` scales after a short queue delay.
- Async video requests use an explicit per-job RunPod policy, because the RunPod default `executionTimeout` is only 10 minutes.

### Why 48GB minimum?
- WAN 2.2 14B Q6_K at 480×848 @ 321 frames needs ~26GB VRAM
- 24GB GPUs (3090, 4090) can't fit the model + activations for long videos
- 48GB gives comfortable headroom for higher resolutions

## API Usage Example

```python
# ✅ CORRECT — architecture tier IDs
gpuIds = "AMPERE_48,ADA_48_PRO,AMPERE_80"

# ❌ WRONG — model names (API accepts but scheduler ignores!)
gpuIds = "NVIDIA GeForce RTX 4090,NVIDIA A100 80GB"
```

### GraphQL Mutation

```graphql
mutation {
  saveEndpoint(input: {
    id: "your-endpoint-id",
    name: "your-endpoint-name",
    gpuIds: "AMPERE_48,ADA_48_PRO,AMPERE_80"
  }) {
    id
    gpuIds
  }
}
```

## Discovery Method

To discover new tier IDs (e.g., if RunPod adds new GPU types):
1. Go to RunPod UI → Serverless → Your Endpoint → Edit
2. Check ALL GPU checkboxes → Save
3. Query the endpoint via API: `{ myself { endpoints { id gpuIds } } }`
4. Read back the `gpuIds` string — it contains all valid tier IDs
5. **Restore your production config** after discovery!
