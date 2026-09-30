# LTX-2.3 → LTX-2.5 migration plan (prepared, not executed)

> **Status: PLAN ONLY.** Nothing in this document has been executed. No deploys, no
> RunPod jobs, no weight downloads, no code changes. Every step is labelled
> **[verified]** (confirmed against a primary source while writing this plan) or
> **[needs runtime confirmation]** (cannot be settled without a GPU run or the
> operator's action).
>
> Prepared 2026-09-30. Scope: the platform's **second video slot**, cloud-only on
> RunPod serverless worker `deploy/runpod-ltx23/` (endpoint `oelala-ltx23`,
> template `c1fz26l07d`).

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Is `Lightricks/LTX-2.5` gated? | **Yes — `gated: "auto"`** (automatic approval). LTX-2.3 is **not** gated. **[verified]** |
| Same license as 2.3? | **No.** 2.5 uses the *LTX-2.x Community License* (dated 2026-08-11); 2.3 uses the *LTX-2 Community License* (dated 2026-01-05). Both permit commercial use below **USD 10,000,000** annual revenue. | **[verified]** |
| What the new license adds | **AI-regulation obligations** — EU AI Act + California AI Transparency Act compliance, no removal of watermarking/provenance features, output disclosure, and indemnity. **Absent from the 2.3 license.** See §3.2.1. | **[verified]** |
| Can our worker image run 2.5 as-is? | **No.** Our image carries **ComfyUI 0.18.1**; LTX-2.5 core support landed in **ComfyUI 0.32.0**. **[verified]** |
| Recommended 2.5 file set (48 GB class) | int8 distilled transformer + int8 LTX-projected Gemma 4 12B + video VAE + audio VAE = **38.71 GB** **[verified]** |
| Cold start vs today | **~60 s faster** than the current 2.3 set (38.71 GB vs 59.57 GB at ~350 MB/s) **[verified sizes; transfer rate is an assumption]** |
| Cost of the acceptance test | **≈ USD 0.10 (A40) – 0.23 (A100)** for one 121-frame test including cold start **[needs runtime confirmation]** |

**Status of the gate: ACCEPTED (2026-09-30) on the HuggingFace account `bomehika`.**
Weight access is confirmed (HTTP 206). **The next blocking item is not the licence but the
token wiring:** the deployed template currently carries `HF_LORA_TOKEN` (m0nk111), which is
**403** on this repo — deploying 2.5 as-is would fail the cold-start download. Fix per §A3
before any GPU spend.

---

## 2. Corrections to the working assumptions

Two assumptions in the task brief do not survive contact with the sources:

1. **"our pinned ComfyUI (likely 0.37.2)"** — wrong. The Dockerfile pins *nothing*;
   it clones ComfyUI at HEAD. The image was built **2026-04-12** (tag
   `20260412-102222`, Dockerfile commit `c820cac`, 2026-04-12 20:14 UTC), and
   ComfyUI HEAD at that moment was commit `31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c`
   — `comfyui_version.py` and `pyproject.toml` at that commit both say **0.18.1**.
   So we are **14 minor versions** behind the 2.5 minimum, not one. (For reference,
   `v0.37.2` is a real ComfyUI tag — commit `830232b8…`, dated **2026-09-23** — but it
   postdates our 2026-04-12 image by five months, so it cannot be what the image
   contains. Note that `v0.37.2` has **no GitHub release**; it exists only as a git
   tag, which is why a `releases/tags/v0.37.2` lookup 404s.) **[verified]**

2. **"the repo's file listing is not readable without accepting the license"** — wrong,
   and this is good news. The HF *metadata* API exposes the full file listing **with
   per-file sizes** to unauthenticated callers even for a gated repo. Only file
   *content* (`/resolve/...`) is blocked (HTTP 401). The entire file table in §4 was
   obtained without accepting anything. **[verified]**

A third fact worth recording because it invalidates a secondary claim in
`docs/LEGACY.md`: **`comfyanonymous/ComfyUI` no longer exists as a path** — GitHub
permanently redirects it to **`Comfy-Org/ComfyUI`**. `git clone` still works through
the redirect, but the Dockerfile URL is stale and should be updated. **[verified]**

---

## 3. Gating and license

### 3.1 Gating status

| Model | API `gated` | Gating UI | `lastModified` |
|---|---|---|---|
| `Lightricks/LTX-2.5` | `"auto"` | Button **"Agree and Access"** | 2026-09-01 |
| `Lightricks/LTX-2.3` | `false` | none | 2026-08-27 |

Evidence:

```bash
curl -s https://huggingface.co/api/models/Lightricks/LTX-2.5 | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['gated'], d['cardData']['extra_gated_button_content'])"
# -> auto Agree and Access
```

`"auto"` means **approval is automatic**: no manual review, no waiting queue. The
gate text (from `cardData.extra_gated_description`) reads:

> By clicking "Agree and Access" you acknowledge the [Privacy Policy](...) and
> consent to receive offers and updates including targeted and personalized
> advertisements. You can unsubscribe at any time.

Content access is genuinely blocked without it: an unauthenticated `HEAD` on
`https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors`
returns **401**. (Note: for a gated repo a *nonexistent* filename also returns 401,
so a 401 on `/resolve/` alone does not prove a file exists — but the metadata API
listing in §4 does.) **[verified]**

### 3.2 License

| | LTX-2.5 | LTX-2.3 |
|---|---|---|
| `license` | `other` | `other` |
| `license_name` | `ltx-2.x-community-license-agreement` | `ltx-2-community-license-agreement` |
| `license_link` | `github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x` | `github.com/Lightricks/LTX-2/blob/main/LICENSE-2` |
| Dated | **August 11, 2026** | **January 5, 2026** |

**These are two different documents**, so the license does change with the migration.
Per the `LICENSE` pointer file: the LTX-2.x agreement applies to "all LTX-2.5 versions
released since August 11, 2026, and all future releases of LTX-2.x", while the LTX-2
agreement covers "LTX-2 versions released since January 5, 2026, **including LTX-2.3
until August 11, 2026**". The 2.5 HF repo itself ships **no license file** (only weights
+ `README.md`), and the model card's licence link is broken (see §11). The license lives
in the `LTX-2` GitHub repo. **[verified]**

Relevant clauses of `LICENSE-2_x` (LTX-2.x Community License, retrieved
2026-09-30, 30 399 bytes):

- **Commercial use below the threshold is permitted, royalty-free.** §2.1 grants a
  "non-exclusive, worldwide, non-transferable and royalty-free limited license …
  to use, reproduce, prepare, distribute, publicly display, publicly perform,
  sublicense, copy, create derivative works of, and make modifications to LTX-2.x,
  for any purpose".
- **Revenue threshold: USD 10,000,000/year.** §2.1: "Entities with annual revenues of
  at least $10,000,000 (the "Commercial Entities") are required to obtain a paid
  license for any use … of LTX-2.x". §1.2/§1.3 aggregate revenue across controlled
  affiliates. A small commercial service below that figure needs **no** paid license.
  **LTX-2.3's `LICENSE-2` carries the identical USD 10 M threshold** — the migration
  does not change our licensing position. **[verified]**
- **Hosted / SaaS use is expressly allowed.** §3: "You may host for third parties
  remote access purposes (e.g. software-as-a-service), reproduce and distribute
  copies … provided that you meet the following conditions".
- **Those conditions are obligations on us.** §3.1: the use-based restrictions and
  all of Attachment A "MUST be included as an enforceable provision by you in any
  type of legal agreement … governing the use and/or distribution of LTX-2.x", and
  §3.2 requires passing the agreement to recipients.
- **No attribution or "powered by" requirement** was found in the license. Section 3.4
  *of the licence* does require retaining "copyright, patent, trademark, and attribution
  notices", and Section 8 forbids using Lightricks trademarks or implying endorsement.
  **[verified]**

### 3.2.1 What legally *changes* when moving 2.3 → 2.5

This is not a cosmetic license refresh. Verified by diffing `LICENSE-2` against
`LICENSE-2_x` (grep counts are decisive here):

| Clause | LTX-2.3 (`LICENSE-2`) | LTX-2.5 (`LICENSE-2_x`) |
|---|---|---|
| USD 10 M revenue threshold | present | present (**unchanged**) |
| Royalty-free grant below threshold | present | present (**unchanged**) |
| SaaS / hosted use allowed | present (§3) | present (§3) |
| AUP incorporated (NSFW restriction) | present | present (**unchanged**) |
| "Non-Commercial Purpose" carve-out (§2.2) | **absent** (0 hits) | **added** (4 hits) |
| Liquidated damages on breach | "**double** the amount" (1 hit) | **removed** (0 hits) — replaced by standard commercial rates |
| §3.5 Transfer of Derivatives | absent | **added** |
| **AI-regulation obligations (§6)** | **absent (0 hits)** | **added** — EU AI Act + California AI Transparency Act |
| EU AI Act Art. 53(2) FOSS statement | absent | **added** |

**The §6 addition is the one that matters most for this platform.** `LICENSE-2_x`
§6 requires that you:

- comply with **Regulation (EU) 2024/1689 (the EU AI Act)** and the **California AI
  Transparency Act** (Cal. Bus. & Prof. Code § 22757 et seq.) as they apply to your
  use, deployment or distribution;
- **not remove, disable, alter or circumvent** any safety measure, disclosure, metadata,
  watermarking, content-provenance or latent-disclosure feature — and **include
  equivalent obligations in your own downstream agreements**;
- accept sole responsibility as a **provider or deployer**, *including any obligation to
  disclose that content is artificially generated or manipulated*;
- indemnify Lightricks against claims arising from your non-compliance.

Licensor may also **revoke the license immediately** if you modify the model such that
it can no longer emit required disclosures. §6 further states: *"You shall undertake
reasonable efforts to use the latest version of LTX-2.x. Any use of the non-current
version of LTX-2.x is done solely at your own risk."*

**[verified: all quotes read directly from the retrieved license texts]**

**Consequences the operator should weigh:**

1. Migrating to 2.5 **adds** AI-transparency obligations that the currently deployed
   2.3 worker does not carry. Output disclosure ("this is AI-generated") becomes a
   licence condition, not merely a product choice.
2. The prohibition on circumventing content filters / provenance features sits
   awkwardly beside an NSFW-capable product — this sharpens the §3.3 question below.
3. The "use the latest version" clause mildly complicates the version-pinning this plan
   recommends (Phase B1/B2). Pinning is still the right engineering call — the clause
   allocates *risk*, it does not forbid pinning — but it is worth recording that
   staying on an old LTX-2.x build is at our own risk under the 2.5 license.

### 3.3 The NSFW point the operator must decide on

Attachment A of `LICENSE-2_x` incorporates the **Lightricks Acceptable Use Policy**
by reference, and that policy contains an explicit section:

> **Do Not Generate Sexually Explicit Content**
> This includes using our Products to: Depict or request sexual intercourse or sex
> acts; Generate content related to sexual fetishes or fantasies; Facilitate,
> promote, or depict incest or bestiality; Engage in erotic chats

**This is not new in 2.5.** `LICENSE-2` (the 2.3 license) incorporates the same
Acceptable Use Policy in its Attachment A. So this exposure already exists on the
currently deployed 2.3 worker; the migration neither creates nor removes it. It is
recorded here as an operator/legal question, not as a migration blocker.
**[verified]**

---

### 3.4 Direct-pull vs mirroring — a good instinct, with three caveats

**Recommended architecture: the worker pulls weights directly from
`Lightricks/LTX-2.5` with the `bomehika` token. Do not mirror the official weights into
our own public repo.** This is the right call and the plan adopts it. It keeps us out of
the redistribution path for the official files and avoids publishing gated weights under
our own name.

Two caveats, because the reasoning "then we are not a distributor and need not enforce
anything" is **too strong**:

**Caveat 1 — §3's trigger phrase is hosting, not copying.** §3 reads: *"You may **host
for third parties remote access purposes (e.g. software-as-a-service)**, reproduce and
distribute copies … provided that you meet the following conditions: 3.1 Use-based
restrictions … MUST be included as an enforceable provision by you in any type of legal
agreement governing the use and/or distribution of LTX-2.x."* Oelala is precisely a
service that hosts model capability for third parties. On the plain reading, **the §3.1
obligation attaches to the hosting itself**, not only to shipping weight files — so
direct-pull may not remove it. Whether "we never hand over the weights" is enough to
escape §3.1 is a **legal judgement, not a technical one**, and it should be confirmed
rather than assumed. The low-risk posture is to put the use restrictions in our ToS
either way; the cost of doing so is small and the downside of being wrong is a material
breach with fee exposure.

**Caveat 2 — some obligations do not depend on distribution at all.**
- **Attachment A** applies to *use*: *"When using the Outputs, LTX-2.x and any
  Derivatives thereof, you agree to comply with the Acceptable Use Policy."* That binds
  us as a user regardless of how the weights arrive.
- **§6** (AI regulations) applies to our *"use, deployment, or distribution"* — deployment
  is enough. The output-disclosure and anti-circumvention duties stand on their own.
- **§3.2** requires passing the agreement to recipients; if we never distribute, this
  simply does not arise.

**Caveat 3 — we already are a distributor for the mirror's contents.** The public mirror
`bomehika/oelala-models` hosts the community **REDGraft checkpoint, which is a
Derivative**. For that file we are distributing, so §3.1 (embed the restrictions),
§3.2 (supply a copy of the agreement) and §3.5 (the recipient gains no additional
rights) apply to it. Adding `README.md` + `LICENSE-LTX-2.x.txt` alongside it is the
correct practice and satisfies §3.2's "provide a copy" condition for that file. The
"we are not a distributor" framing therefore does not hold globally — only for the
official files we choose not to mirror.

**Net recommendation:** pull official weights directly (as proposed), keep the licence
text with anything we do mirror, and **still put the use restrictions in our own ToS** —
that is the position that is defensible under every reading. **[verified: all quoted
clause text; the legal conclusion is flagged for review, not asserted]**

Sizes below are **bytes from the HuggingFace API** (`?blobs=true`), converted as
GB = 10⁹ bytes. Retrieved unauthenticated on 2026-09-30. **[verified]**

## 4. Exact file set for a 48 GB-class worker

### 4.1 `Lightricks/LTX-2.5` — complete listing (17 files, 200.85 GB)

| Path | GB | Purpose |
|---|---:|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42.018 | distilled transformer, bf16 |
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42.018 | dev transformer, bf16 |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | **21.504** | distilled transformer, ComfyUI-native int8 |
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | 21.504 | dev transformer, ComfyUI-native int8 |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | **18.722** | distilled transformer, nvfp4 |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26.264 | LTX-projected Gemma 4 12B text encoder, bf16 |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | **15.373** | LTX-projected Gemma 4 12B, ComfyUI-native int8 |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | **1.472** | video VAE |
| `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1.452 | video VAE, conv variant |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | **0.365** | audio VAE |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8.900 | distilled LoRA (two-stage / dev paths) |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 0.996 | 2× spatial upscaler (two-stage only) |
| `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | 0.262 | temporal upscaler (two-stage only) |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 0.004 | duration head |
| `.gitattributes`, `README.md`, `hf-hero-web.webp` | ~0 | — |

**Two sibling repos exist but are not usable here** (both also `gated: auto`):

| Repo | Contents | Why not |
|---|---|---|
| `Lightricks/LTX-2.5-Diffusers` | 66 files, diffusers layout (`transformer/diffusion_pytorch_model-0000N-of-00004.safetensors`, `connectors/…`) | not ComfyUI-native — our worker loads safetensors via `UNETLoader` |
| `Lightricks/LTX-2.5-Pre-Trained` | 10 files, `ltx-2.5-22b-pt-bf16.safetensors` 46.17 GB, `ltx-2.5-22b-gemma4-12b/model.safetensors` 23.92 GB | fine-tuning base, not an inference checkpoint |

Also confirmed absent: `Lightricks/LTX-2.5-{fp8,nvfp4,distilled}` — 2.5 ships its
quantizations **inside** the main repo (unlike 2.3, which has separate `-fp8` and
`-nvfp4` repos). **[verified]**

### 4.2 Recommended set — distilled int8 I2V + T2V, 48 GB class

This is the minimal set that satisfies the official single-stage distilled workflow
(§6). **Bold** files are the four that actually load.

| File | Repo | GB |
|---|---|---:|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` | 21.504 |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` | 15.373 |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | `Lightricks/LTX-2.5` | 1.472 |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | `Lightricks/LTX-2.5` | 0.365 |
| **Total** | | **38.714** |

**Do not substitute `Comfy-Org/gemma-4`'s `gemma4_12b_int8_convrot.safetensors`
(12.055 GB) for the LTX encoder.** That repo is ungated and tempting, but the file
lacks the LTX projection (`with-proj-ltx-2.5`); the official workflow loads
`gemma4-12b-with-proj-ltx-2.5-*` through a `CLIPLoader` with type `ltxv`. Using the
plain Gemma 4 encoder is expected to fail or produce garbage.
**[verified that the two files differ and that the workflow names the `with-proj` one;
needs runtime confirmation that the plain one actually fails]**

**This set is not our inference — it is the vendor's own default.** ComfyUI's built-in
LTX-2.5 template (`Comfy-Org/workflow_templates`, `templates/video_ltx2_5_t2v.json`)
loads:

| Node | Value |
|---|---|
| `UNETLoader` | `ltx-2.5-22b-distilled-transformer-**comfy-int8-convrot**.safetensors` |
| `CLIPLoader` | `gemma4-12b-with-proj-ltx-2.5-**comfy-int8-convrot**.safetensors`, type `ltxv` |
| `VAELoader` | `ltx-2.5-video-vae-bf16.safetensors` |
| `VAELoader` | `ltx-2.5-audio-vae-bf16.safetensors` |
| `CLIPLoader` (enhancer, optional) | `gemma4_e2b_it_int8_convrot.safetensors`, type `ltxv` |

**The int8 path is the official default**, which is a strong confirmation that §4.2 is
the right target for a 48 GB-class worker. **[verified]**

**One qualification: that built-in template is *two-stage*.** It carries
`LatentUpscaleModelLoader` + `LTXVLatentUpsampler`, two `ManualSigmas`, two
`SamplerCustomAdvanced` and two `LTXVDualCFGGuider` nodes, and loads the
`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (0.996 GB). So the faithful
"official default" set is **39.710 GB**, while our minimal single-stage target — which
mirrors the pack's `LTX-2.5_T2V_I2V_Single_Stage_Distilled.json`, the closer analogue of
our existing 2.3 builders — is **38.714 GB**. Both are official; the 1 GB difference is
the upscaler. Pick per the single-vs-two-stage decision in §10.

**One non-issue, for the record:** the audio VAE's HF path is
`LTX2_audio_vae_bf16.safetensors` (capital L/T) while our registry stores it locally as
`ltx2_audio_vae.safetensors`. That is **correct as written** — `handler.py` pairs
`"filename": "ltx2_audio_vae.safetensors"` with
`"hf_path": "LTX2_audio_vae_bf16.safetensors"`, and `hf_hub_download` is called with the
`hf_path`. No change needed. **[verified]**

**There is no official ungated Comfy-Org mirror of LTX-2.5.** `Comfy-Org/LTX-2.5`
does not resolve (401). Comfy-Org publishes the 2.3-era repacks only — `Comfy-Org/ltx-2`
(9 files) and `Comfy-Org/ltx-2.3` (6 files), both ungated — and an HF-wide search for
`LTX-2.5` returns only the Lightricks repos plus third-party community quantizations
(GGUF/FP8/MLX by unrelated accounts). **Consequence: the gated `Lightricks/LTX-2.5`
repo is the only official source, so the gate is unavoidable for a sanctioned set.**
**[verified]**

### 4.3 The nvfp4 variant — and why int8 is the right call for 48 GB

`diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` — **18.722 GB**,
the smallest transformer available. With the int8 text encoder the set totals
**35.932 GB**.

**nvfp4 is a dead end for the platform's 48 GB tiers.** ComfyUI gates it on compute
capability, verified in `comfy/model_management.py` at v0.38.0:

```python
def supports_nvfp4_compute(device=None):
    if not is_nvidia():
        return False
    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        return False
    return True
```

and `comfy/ops.py::get_disabled_quant_formats()` adds `"nvfp4"` to the disabled set
when that check fails. `major < 10` excludes **A40 (8.6), A100 (8.0), L40S (8.9)** and
every other Ampere/Ada tier — only Blackwell (CC 10.x+) qualifies. On an unsupported
device the format is **disabled rather than used**, so the model still loads correctly
but you get **none** of the speed or memory benefit of the quantization. **[verified:
both functions read directly from v0.38.0 source]**

This matches the hardware documentation independently: native FP4 tensor-core support
appears at compute capability 10.x/11.0/12.x and is absent at 8.0/8.6/9.0 (CUDA
Programming Guide, compute-capabilities table; TensorRT-LLM's quantization hardware
matrix marks Blackwell `Y` and Hopper/Ada/Ampere `.`). Some community documentation
describes the fallback as "silently dequantizing"; the source shows an explicit
capability check that disables the format, so the practical outcome is the same but the
mechanism is deliberate, not silent. **[verified: source is authoritative here]**

**By contrast, int8 has no such gate** — and this is what makes the recommended set
safe:

```python
def supports_int8_compute(device=None):
    # ... returns False only for MPS, Intel XPU, DirectML and ixuca
    return True          # any NVIDIA CUDA device
```

`int8_tensorwise` (which is what a `comfy-int8-convrot` file loads as — `convrot` is a
flag on that format, not a separate one) is therefore available on A40, A100 and L40S
alike. **[verified]**

**Conclusion:** use int8-convrot on the current 48 GB tiers; consider nvfp4 only if the
platform standardises on a Blackwell tier (`BLACKWELL_96` / `BLACKWELL_180`). The
~3 GB saving is not worth a hardware migration on its own.

### 4.4 The community-checkpoint shortcut — technically real, legally not a shortcut

The public mirror `bomehika/oelala-models` (**ungated**, 9 files, last modified
2026-09-30) contains a community LTX-2.5 int8 checkpoint:

| Path | GB |
|---|---:|
| `diffusion_models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors` | **17.026** |
| `LICENSE-LTX-2.x.txt` | ~0 |

The mirror also ships the LTX-2.x licence text alongside the weights. **[verified:
repo exists, ungated, file and size read from the HF API. The measured size is
17.026 GB — not 16.63 GB as reported informally.]**

**This solves the access problem but NOT the licence question, and the difference
matters.** The LTX-2.x licence grants rights over "LTX-2.x **and Derivatives of
LTX-2.x**" (§2.1), and §1.5 defines Derivatives to include modified weights. A
community int8 quant is therefore a Derivative, and:

- **§2.1** — the USD 10 M revenue threshold applies to "LTX-2.x **and Derivatives of
  LTX-2.x**", so the paid-licence obligation is unchanged by where you downloaded it.
- **§3.5 (Transfer of Derivatives)** — a Derivative must be distributed under the terms
  of this same agreement, which is why the mirror carries `LICENSE-LTX-2.x.txt`.
- **Attachment A** — the use restrictions and the Acceptable Use Policy apply to "the
  Outputs, LTX-2.x **and any Derivatives thereof**".

**So: downloading from an ungated mirror does not remove the obligation to comply with
the LTX-2.x Community Licence.** It only removes the click. The operator still has to
honour the licence (and the AUP) whether the weights come from the gated repo or from
a third-party quant of them. **[verified: licence text; the practical consequence is a
legal judgement, not a technical one]**

Two further caveats before treating this file as a drop-in:

- **Provenance and quality are unverified.** It is a third-party Civitai quantization
  (REDGraft "Fast 2K" int8). Its tensor layout, quant format and prompt-following
  behaviour have not been validated against the official int8 file. It may not load via
  `UNETLoader` at all, and it is not the file the official workflows were tuned for.
  **[needs runtime confirmation]**
- **The text encoder stays gated.** Even with an ungated transformer, the
  `gemma4-12b-with-proj-ltx-2.5-*` encoder has no ungated official equivalent (§4.2).
  Without gate acceptance there is **no complete 2.5 set** — the operator's own
  assessment is correct on this point.

**Recommendation:** treat the community checkpoint as a *fallback for experimentation*,
not as the production path. If the gate is accepted (Phase A), use the official int8
files; they are the ones the vendor's workflows target and the ones whose licence
position is unambiguous.

### 4.5 Cold-start download size

| Set | Total GB | Cold start @ ~350 MB/s |
|---|---:|---:|
| **LTX-2.3 today** (`ltx-2.3-22b-distilled` 46.149 + `gemma_3_12B_it_fp8_scaled` 13.205 + `LTX2_audio_vae_bf16` 0.218) | **59.572** | **170 s** |
| **LTX-2.5 int8 (recommended)** | **38.714** | **111 s** |
| LTX-2.5 int8 + spatial upscaler (official built-in two-stage default) | 39.710 | 113 s |
| LTX-2.5 int8 + both upscalers (two-stage, temporal optional) | 39.972 | 114 s |
| LTX-2.5 nvfp4 + int8 TE | 35.932 | 103 s |
| LTX-2.5 bf16 distilled | 70.119 | 200 s |

The 2.3 registry's own `size_gb` values (46.1 / 13.2 / 0.22, summing to **59.52 GB**)
match the measured API sizes to within ~1 % (46.149 / 13.205 / 0.218), so the registry
is accurate today and the same convention can be used for the 2.5 entries.
**[verified]**

**The recommended 2.5 set is ~21 GB smaller than today's**, i.e. roughly **60 s
faster** to cold-start. The bf16 2.5 path would be ~30 s *slower* than today. The
350 MB/s figure is the brief's assumption and is **not** verified — treat the seconds
as proportional, not absolute. **[sizes verified; rate assumption not verified]**

---

## 5. Worker compatibility — can our image run 2.5?

### 5.1 What our image actually contains

`deploy/runpod-ltx23/Dockerfile` pins **nothing**:

```dockerfile
FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .      # HEAD, unpinned
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git  # HEAD
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git    # HEAD
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git          # HEAD
```

| Component | Version in the deployed image | Evidence |
|---|---|---|
| ComfyUI | **0.18.1** | HEAD at 2026-04-12 = `31283d28…`; `comfyui_version.py` **and** `pyproject.toml` both `0.18.1` |
| ComfyUI-LTXVideo | **`531512f728…` (2026-03-06)** | last pack commit before the image build; the pack syncs on a `pr-YYYY-MM-DD` bot branch, so HEAD was a month old at build time |
| VHS / KJNodes | HEAD around 2026-04-12 (not recorded) | cloned at HEAD, never pinned |
| torch | 2.9.1 (cu130) | base image tag |

**Answer: NO — our image cannot run LTX-2.5 as-is, and this is now *observed at the code
level*, not inferred from release notes.** Probed at our exact build commit
`31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c`:

```bash
C=31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/$C/comfy/text_encoders/gemma4.py"
# -> 404          (the file does not exist at all)
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/$C/comfy/ldm/lightricks/duration_head.py"
# -> 404
curl -s "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/$C/comfy/sd.py" | grep -c gemma4
# -> 0            (v0.32.0: 21)
```

**LTX-2.5's text encoder is Gemma-4-based, and our build has no Gemma-4 text-encoder
module and no `gemma4` path in `sd.py` at all.** The conditioning path for 2.5 simply
does not exist in the image. **[verified — file existence and symbol counts, both
observed]**

A second gap: seven pack nodes the 2.5 graphs use were not in the 2026-03-06 pack either
— `LTXFloatToInt`, `LTXVAudioOnlyModel`, `LTXVAudioOnlyEmptyVideoLatent`,
`LTXVGetTilingSizes`, `LTXVTiledFusionSampler`, `LTXVLaplacianPyramidBlend`,
`LTXVSetAudioRefTokens`. (Only `LTXFloatToInt` matters for the single-stage path we
target; the others serve T2A, tiled-fusion and audio-reference workflows.)
**[verified]**

### 5.2 Minimum versions required for 2.5

| Component | Minimum | Evidence |
|---|---|---|
| **ComfyUI core** | **0.32.0** (2026-08-11) | release notes: *"Add support for LTX 2.5 by @alexisrolland in PR #15499"* and *"[Partner Nodes] feat(LTX): add new nodes for model version 2.5"* |
| ComfyUI core (official 2.5 blueprints) | 0.35.0 (2026-09-09) | `blueprints/Image to Video (LTX-2.5).json` first added in `54e03f53…`, where `comfyui_version.py` = `0.35.0` |
| **ComfyUI-LTXVideo** | **no version exists** — pin a commit | the pack has **no tags and no releases**; `pyproject.toml`, `setup.py` and `version.py` all 404 |
| PyTorch | ≥ 2.7 | v0.32.0 notes: *"Minimum officially supported pytorch is now 2.7"* — our base image has 2.9.1, so **this is satisfied** |

**The 0.32.0 floor is observed, not just inferred from release notes.** Checking the
release immediately *before* it pins the boundary by code inspection:

```bash
for T in v0.31.0 v0.32.0; do
  echo -n "$T LTXVDualCFGGuider: "
  curl -s https://raw.githubusercontent.com/Comfy-Org/ComfyUI/$T/comfy_extras/nodes_lt.py | grep -c LTXVDualCFGGuider
  echo -n "$T gemma4_text_encoder_model: "
  curl -s https://raw.githubusercontent.com/Comfy-Org/ComfyUI/$T/comfy/text_encoders/gemma4.py | grep -c gemma4_text_encoder_model
done
# v0.31.0 LTXVDualCFGGuider: 0        v0.32.0 LTXVDualCFGGuider: 3
# v0.31.0 gemma4_text_encoder_model: 0 v0.32.0 gemma4_text_encoder_model: 1
```

v0.31.0 ships Gemma 4 generally but has **no code path for the LTX-projected
`gemma4-12b-with-proj-ltx-2.5` encoder**, and lacks the 2.5 guiders. **[verified]**

Recommended pins for the migration:

| Component | Pin | Note |
|---|---|---|
| ComfyUI | tag **v0.38.1** = `20ca544ee0436721d8eb5f544665e490609f72c8` (2026-09-30) | latest tag; 0.32.0 is the floor |
| ComfyUI (fallback) | tag **v0.38.0** = `6b747c0428c343e1417219641db93a4fb7cb69ae` (2026-09-29) | the release with published notes |
| ComfyUI-LTXVideo | commit **`bf2ca0264f706db64cb8931155695ca481fc9d91`** (master, 2026-09-30) | latest |
| ComfyUI-LTXVideo (minimum) | commit **`9d1672beb6c606cde771228927044b78ecc7db8a`** (2026-08-11) | first commit adding `example_workflows/2.5` |
| ComfyUI-LTXVideo (currently in our image) | commit **`531512f7286963dc7aff1fd8bf5556e95eae03af`** (2026-03-06) | last sync before our 2026-04-12 build |

**[verified]** Note the release/tag asymmetry in this repo: many ComfyUI versions exist
as git tags **without** a GitHub release (v0.37.2, v0.38.1), so verify a pin by tag
lookup rather than by release lookup.

**[verified]** The minimum pack commit is verified by history: `9d1672beb6…`
("Automated PR - 2026-08-11") is the **first** commit that touches
`example_workflows/2.5`. Note that `docs/LEGACY.md` refers to "code v1.4.x" for the
LTX pack — that version string **could not be found anywhere** in the pack (no tags,
no version file, no `pyproject.toml`). Treat "v1.4.x" as **UNVERIFIED**.

**VHS and KJNodes are not required by the 2.5 workflows** — they are pulled in for our
2.3 path (video output via VHS, `ImageResizeKJv2` for I2V input resize). Keep them,
but do not expect the 2.5 migration to need new third-party node packs.
**[verified for the workflows analysed; needs runtime confirmation that our own I2V
resize path still works]**

### 5.3 Which node classes are actually missing

I extracted the node classes from the official 2.5 single-stage distilled workflow
(`example_workflows/2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json`, subgraphs
flattened) and checked each against the pack and against core at 0.18.1.

> **Pin the right thing.** Eight of the nine LTX nodes our 2.3 builders use are
> **ComfyUI *core*** nodes (`comfy_extras/nodes_lt.py`, `nodes_lt_audio.py`), not
> LTXVideo-pack nodes. Pinning the pack to a commit does **not** pin them — they move
> with ComfyUI itself. The pack only owns `LTXVImgToVideoConditionOnly`, `LTXFloatToInt`,
> `MultimodalGuider`, `GuiderParameters`, `GemmaAPITextEncode` and friends. This is why
> the ComfyUI core pin (Phase B1) is the load-bearing one. **[verified]**

| Node class | In pack (latest)? | In core 0.18.1? | Needed by 2.5? |
|---|---|---|---|
| `LTXVImgToVideoInplace` | no | **yes** (`comfy_extras/nodes_lt.py`) | **yes** |
| `LTXVPreprocess` | no | yes | yes |
| `LTXVConditioning` | no | yes | yes |
| `LTXVConcatAVLatent` / `LTXVSeparateAVLatent` | no | yes | yes |
| `LTXVEmptyLatentAudio` / `LTXVAudioVAEDecode` | no | yes | yes |
| `EmptyLTXVLatentVideo` | no | yes | yes |
| `UNETLoader`, `CLIPLoader`, `VAELoader` | no | yes | yes |
| `CFGGuider`, `KSamplerSelect`, `ManualSigmas`, `RandomNoise`, `SamplerCustomAdvanced` | no | yes | yes |
| `VAEDecodeTiled`, `CreateVideo`, `SaveVideo` | no | yes | yes |
| `LTXFloatToInt` | **yes** | no | yes |
| `GemmaAPITextEncode` | **yes** | no | only for the API prompt enhancer |
| `TextGenerateLTX2Prompt` | no | not found at 0.18.1 | only for the prompt enhancer |
| `ComfySwitchNode`, `StringContains` | no | yes | graph plumbing |
| `LTXVImgToVideoConditionOnly` (our 2.3 node) | **yes — still present** | no | not used by 2.5 |
| `MultimodalGuider` / `GuiderParameters` (our 2.3 AV nodes) | **yes — still present** | no | not used by the 2.5 single-stage graph |

**The surprise: the node-class gap is much smaller than expected.** Almost everything
the 2.5 workflow needs is a *core* node that already exists in 0.18.1, and the pack
still ships our 2.3 nodes. The blocking issue is **not** missing node classes — it is
that **ComfyUI 0.18.1's `comfy/ldm/lightricks/` cannot load the 2.5 architecture**,
which is precisely what PR #15499 ("Add support for LTX 2.5") delivered in 0.32.0.
**[verified: node inventories; needs runtime confirmation that 0.18.1 fails on 2.5
weights — the failure is inferred from the release notes, not observed]**

A secondary structural risk: the official 2.5 workflows are **subgraph-based**
(`definitions.subgraphs`, with subgraph UUIDs as node types), whereas our builders emit
a **flat API-format graph**. Our builders do not need to become subgraphs — the API
format is what `SamplerCustomAdvanced` etc. consume — but we cannot copy the official
JSON verbatim. **[verified]**

---

## 6. Workflow impact

Source: the official
`example_workflows/2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json`
(153 765 bytes, 18 top-level nodes, 5 subgraphs, 25 links), read directly from the
pack on master. **[verified]**

### 6.1 What the official 2.5 graph does, exactly

**Load Models** subgraph:

| Node | Title | Widget values |
|---|---|---|
| `UNETLoader` | — | `ltx-2.5-22b-distilled-transformer-bf16.safetensors`, `default` |
| `VAELoader` | Load VAE - Video | `ltx-2.5-video-vae-bf16.safetensors` |
| `VAELoader` | Load VAE - Audio | `ltx-2.5-audio-vae-bf16.safetensors` |
| `CLIPLoader` | Load CLIP - Text Encoder | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, `ltxv`, `default` |
| `CLIPLoader` | Load CLIP - Text Enhancer | `gemma4_e2b_it_bf16.safetensors`, `ltxv`, `default` |

**Sampler - Distilled (8 steps)** subgraph:

| Node | Widget values |
|---|---|
| `CFGGuider` | `cfg = 1` |
| `KSamplerSelect` | `euler_ancestral` |
| `ManualSigmas` | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` |
| `SamplerCustomAdvanced` | — |

**Preprocess** subgraph: `EmptyLTXVLatentVideo` `[960, 544, 121, 1]`,
`LTXVEmptyLatentAudio` `[121, 25, 1]`, `LTXVPreprocess` `img_compression = 18`,
`LTXVImgToVideoInplace` `[strength 0.7, use_image false]`, `LTXVConcatAVLatent`.

**Decode** subgraph: `VAEDecodeTiled` `[512, 64, 64, 8]`, `LTXVAudioVAEDecode`.

### 6.2 The good news: sampling is unchanged

**The distilled sigma schedule, the sampler, CFG=1 and the 8-step count are byte-identical
to what our 2.3 builders already emit.** The `ManualSigmas` string in our
`build_cloud_ltx23_t2v_workflow` and `build_cloud_ltx23_i2v_workflow` matches the
official 2.5 value exactly. **[verified]** Frame-count rule is also unchanged
(`1 + multiple of 8`).

### 6.3 What must change in `src/backend/comfyui_client.py`

| # | Change | Detail | Status |
|---|---|---|---|
| W1 | Replace `CheckpointLoaderSimple` with `UNETLoader` | 2.5 ships the transformer as a standalone file, not a combined checkpoint. Our node 1 must become `UNETLoader(unet_name=…)`. | **[verified]** |
| W2 | Replace `LTXAVTextEncoderLoader` with `CLIPLoader` | `CLIPLoader(clip_name="gemma4-12b-with-proj-ltx-2.5-…", type="ltxv", device="default")`. The 2.3 node combined encoder+checkpoint; 2.5 separates them. | **[verified]** |
| W3 | Add a second `VAELoader` for the video VAE | 2.3 took the video VAE from the checkpoint (node `["1", 2]`); 2.5 needs an explicit `VAELoader(ltx-2.5-video-vae-bf16.safetensors)`. Audio VAE moves from `LTXVAudioVAELoader` to plain `VAELoader`. | **[verified]** |
| W4 | I2V: `LTXVImgToVideoConditionOnly` → `LTXVImgToVideoInplace` | New widget `use_image` (bool) plus `strength` (0.7 in the official graph). Our I2V builder currently passes `strength` only. | **[verified]** |
| W5 | `LTXVPreprocess` `img_compression`: 35 → **18** | Our builders use 35; the official 2.5 single-stage graph uses 18. | **[verified]** |
| W6 | Drop `MultimodalGuider` + `GuiderParameters` for the AV path | The 2.5 single-stage distilled graph uses plain `CFGGuider` even with audio — audio rides along through `LTXVConcatAVLatent`/`LTXVSeparateAVLatent`. Our conditional AV branch (nodes 20-27) simplifies. | **[verified for this graph; needs runtime confirmation for our AV quality expectations]** |
| W7 | `VAEDecodeTiled` widget order | Official `[512, 64, 64, 8]` = `tile_size, overlap, temporal_size, temporal_overlap`. Our builder uses `tile_size 512, overlap 64, temporal_size 512, temporal_overlap 64` — temporal values differ substantially. | **[verified values; effect needs runtime confirmation]** |
| W8 | Optional: prompt enhancer | Official graph loads `gemma4_e2b_it_bf16.safetensors` (10.279 GB, from ungated `Comfy-Org/gemma-4`) and `TextGenerateLTX2Prompt`. **Skip it** for v1 — it adds 10 GB to cold start and a node class absent at 0.18.1. | **[verified]** |
| W9 | Model filename constants | `checkpoint` default → `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`; `text_encoder` default → `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`; `audio_vae_checkpoint` → `ltx-2.5-audio-vae-bf16.safetensors`. | **[verified]** |
| W10 | New builder methods | Add `build_cloud_ltx25_t2v_workflow` / `_i2v_workflow` rather than mutating the 2.3 builders — rollback stays trivial. | **[verified as a design choice]** |

**Marked as needing verification against a real workflow file:** whether the int8
`comfy-int8-convrot` variants load through `UNETLoader`/`CLIPLoader` without an extra
node (the naming strongly implies ComfyUI-native int8, but I could not find
documentation confirming the loader path). **[needs runtime confirmation]**

### 6.4 Caution: the official 2.5 sources disagree — copy a graph, do not synthesise

Two things worth knowing before editing the builders, because they are easy to get
wrong:

1. **There is more than one guider, and two official 2.5 graphs disagree.** The pack's
   single-stage distilled graph uses plain **`CFGGuider` (`cfg = 1`)** — what we already
   emit. ComfyUI's built-in 2.5 template (`Comfy-Org/workflow_templates`,
   `templates/video_ltx2_5_t2v.json`) uses **`LTXVDualCFGGuider` (`[1, 1]`) twice**, once
   per stage. `LTXVDualCFGGuider` is verified present in `comfy_extras/nodes_lt.py` at
   v0.32.0 (3 occurrences) and absent at v0.31.0. **Pick one graph and copy it
   faithfully**; do not mix guider and sigma values across sources.
   **[verified — both graphs parsed directly]**
2. **Stage-2 sigmas genuinely differ between the official sources — and this is now
   settled, not merely observed.** The built-in template's stage 2 uses
   **`0.85, 0.7250, 0.4219, 0.0`** (identical to 2.3), while the pack's
   `2.5/…Two_Stage_Distilled.json` uses **`0.909375, 0.725, 0.421875, 0.0`**. The pack
   README's note *"I2V stage-2 distilled sigmas should be `0.909375, …` (not `0.85, …`)"*
   is therefore **I2V-specific** — the built-in **T2V** template legitimately keeps
   `0.85`. **Do not bulk-rewrite stage-2 sigmas during migration**, and treat 2.5-T2V and
   2.5-I2V as different cases. Stage-1 (the 8-step schedule) is byte-identical across all
   three graphs (`1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`).
   **[verified — all three graphs parsed directly]**
3. **The workflow JSONs carry stale 2.3-era sample prompts.** The 2.5 graph's
   `GemmaAPITextEncode` widgets still reference `ltx-2.3-22b-dev.safetensors` — a
   leftover placeholder, not a required model. Do not treat filenames found inside
   prompt/enhancer widgets as the model manifest; use the loader nodes (§6.1) as the
   authoritative list. **[verified]**
4. **Do not trust community/Civitai workflow configs as the model manifest.** A
   widely-circulated community config for LTX-2.5 reportedly loads a separate vocoder
   (`ltx-av-step-1751000_vocoder_24K`) and a text projection
   (`ltx-2.3_text_projection_bf16`). **Neither belongs to the official 2.5 ComfyUI
   path.** Evidence:
   - The official 2.5 single-stage graph contains **0** occurrences of `vocoder`,
     `step-1751000` or `text_projection`; its entire Load Models subgraph is the five
     nodes in §6.1.
   - `LTXVAudioVAEDecode` (`comfy_extras/nodes_lt_audio.py`, v0.38.0) takes only
     `samples` + `audio_vae` and decodes via `audio_vae.first_stage_model`, reading
     `output_sample_rate` from the VAE itself — **the vocoder lives inside the audio
     VAE**, so no separate vocoder file is needed.
   - `ltx-2.3_text_projection_bf16` is a **2.3-era** filename that exists only in
     scattered single-file user repos, not in `Lightricks/LTX-2.5`; the 2.5 encoder
     ships its projection *inside* `gemma4-12b-with-proj-ltx-2.5-*` (the `with-proj` in
     the name). No separate projection file is required.
   - `ltx-av-step-1751000_vocoder_24K` could not be found on HuggingFace at all.

   **Net effect:** the official 2.5 audio path costs **0.365 GB** (the audio VAE), not
   an extra vocoder + projection. Adding those files would inflate the cold start for
   no benefit. **[verified]**

---

## 7. Migration plan

Order matters: nothing below step 4 costs money, and nothing before step 6 touches
production.

### Phase A — operator actions (no code, no GPU)

> **STATUS 2026-09-30: A1 is DONE.** The licence has been accepted, on the HuggingFace
> account **`bomehika`**. Access was confirmed with ranged requests against
> `Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`:
> **`HF_PUBLIC_TOKEN` (bomehika) → HTTP 206 ✓**, **`HF_LORA_TOKEN` (m0nk111) → HTTP 403 ✗**.
> This empirically confirms that HF grants gated access **per user, not per organisation**.
> **[operator-reported; consistent with the HF gated-models documentation]**

| Step | Action | Status |
|---|---|---|
| **A1** | ~~Accept the gate~~ — **DONE on `bomehika`.** | **[done]** |
| **A2** | Confirm the acceptance took: the model page's gate banner is gone and weight files download. | **[done — verified via 206 ranged request]** |
| **A3** | **⛔ HARD PRECONDITION — the worker template must carry the `bomehika` token, and today it does not.** See the defect below. | **[verified defect; fix required]** |
| **A4** | **Decide on the NSFW question** (§3.3). The AUP prohibits sexually explicit content and the licence requires us to make it enforceable in our own terms. Pre-existing on 2.3, but a migration is the natural moment to settle it. | **[verified: policy text; decision is the operator's]** |
| **A5** | **Decide on AI-transparency compliance** (§3.2.1). New obligation introduced by the 2.5 licence: output disclosure, no circumvention of provenance features, and equivalent clauses in our own ToS. | **[verified: licence text; implementation is the operator's]** |
| **A6** | Confirm the target GPU tiers. Recommended 2.5 int8 set needs ~39 GB with the text encoder resident; an A40 48 GB is tight, so keep the 2.3 worker alive (§9). | **[needs runtime confirmation]** |

#### ⛔ A3 in detail — the deployed template carries the *wrong* token

`deploy/runpod-ltx23/deploy.sh` lines 136-140 hardwire the template's `HF_TOKEN` to
`HF_LORA_TOKEN`:

```bash
HF_LORA_TOKEN=$(grep -E '^HF_LORA_TOKEN=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
if [[ -z "${HF_LORA_TOKEN:-}" ]]; then
    HF_ENV_STR=""
else
    HF_ENV_STR="{ key: \"HF_TOKEN\", value: \"${HF_LORA_TOKEN}\" },"
fi
```

`HF_LORA_TOKEN` belongs to **m0nk111**, which the operator measured as **403 (no gate
access)** on LTX-2.5. **Deploying a 2.5 worker with the current script would therefore
fail its cold-start download with HTTP 403** — not a hypothetical, a defect already in
the repository. It has gone unnoticed because LTX-2.3 is ungated, so that token works
there.

**Required change (Phase B):** point the template's `HF_TOKEN` at `HF_PUBLIC_TOKEN`
(`bomehika`) — either by editing `deploy.sh` to read the right variable, or by accepting
the gate on m0nk111 as well. Note this is a **deployment-script change, not a code
change**, and it is the kind of thing that must not be discovered during a paid GPU run.
**[verified]**

**Do not log the token.** `deploy.sh` interpolates it into a Python heredoc and a GraphQL
mutation; ensure the value never reaches the console or the RunPod job logs.

### Phase B — image and registry (no GPU spend)

| Step | Change | File | Status |
|---|---|---|---|
| **B1** | Pin ComfyUI to a tag (`v0.38.1` = `20ca544e…`, or `v0.38.0` = `6b747c04…`) instead of cloning HEAD; update the stale clone URL to `Comfy-Org/ComfyUI`. | `deploy/runpod-ltx23/Dockerfile` | **[verified]** |
| **B2** | Pin `ComfyUI-LTXVideo` to `bf2ca026…` (no tags exist — commit pin is the only option). | `Dockerfile` | **[verified]** |
| **B3** | Create `models/diffusion_models` — **currently missing**. `NODE_MANAGED_MODEL_DIRS` and the Dockerfile `mkdir` only cover `checkpoints, loras, text_encoders, vae`. ComfyUI resolves `diffusion_models` from `models/unet` or `models/diffusion_models`; neither exists. **Without this the `UNETLoader` step fails.** | `Dockerfile`, `handler.py` | **[verified]** |
| **B4** | Add `diffusion_models` (and `latent_upscale_models` if two-stage is ever wanted) to `NODE_MANAGED_MODEL_DIRS` so volume symlinks work. | `handler.py` | **[verified]** |
| **B5** | Add a new registry entry (do **not** overwrite `LTX23_MODELS`) — e.g. `LTX25_MODELS` with the four files from §4.2, correct `size_gb` (21.5 / 15.4 / 1.5 / 0.37), correct `target_dir` (`diffusion_models`, `text_encoders`, `vae`, `vae`), `repo: "Lightricks/LTX-2.5"`, `startup_required: True`. Accurate `size_gb` matters: `_check_download_capacity()` gates on it. | `handler.py` | **[verified]** |
| **B6** | Confirm `PUBLIC_MODEL_FILENAMES` is derived from the registry (it is: `{m["filename"] for m in LTX23_MODELS}`) so volume assets are not shadowed. | `handler.py` | **[verified]** |
| **B7** | Pin `huggingface_hub` in the Dockerfile (currently unpinned `pip install runpod requests httpx huggingface_hub`). Note the handler passes the deprecated `local_dir_use_symlinks=False` kwarg, which newer `huggingface_hub` versions warn about or reject — pin a known-good version and verify the download path. | `Dockerfile`, `handler.py` | **[verified: unpinned + deprecated kwarg; exact breakage needs runtime confirmation]** |
| **B8** | Build the image with a new dated tag; do **not** push `:latest`. Use `deploy.sh` (it updates the template consistently). | `deploy/runpod-ltx23/deploy.sh` | **[verified: documented in the Dockerfile header]** |
| **B9** | **⛔ Fix the token wiring before deploying 2.5** (§A3): `deploy.sh` lines 136-140 set the template's `HF_TOKEN` from `HF_LORA_TOKEN` (m0nk111 → **403** on LTX-2.5). Point it at `HF_PUBLIC_TOKEN` (bomehika), or accept the gate on m0nk111 too. Also ensure the token is never echoed into console output or job logs. | `deploy/runpod-ltx23/deploy.sh` | **[verified defect]** |

### Phase C — workflows and adapters (no GPU spend)

| Step | Change | File | Status |
|---|---|---|---|
| **C1** | Add `build_cloud_ltx25_t2v_workflow()` per §6.3 (W1-W7, W9). | `src/backend/comfyui_client.py` | **[verified against official workflow]** |
| **C2** | Add `build_cloud_ltx25_i2v_workflow()` per §6.3. | `src/backend/comfyui_client.py` | **[verified against official workflow]** |
| **C3** | Add a `T2V_GENERATION_MODES["ltx25"]` entry (currently only `wan22`, `ltx2` — dated 2026-10-01: `wan22` has since been removed, the family was retired; see `docs/LEGACY.md`). | `comfyui_client.py` | **[verified]** |
| **C4** | Add adapters `ltx25_t2v.py` / `ltx25_i2v.py` (clone the 2.3 adapters; new `name`, `RUNPOD_LTX25_ENDPOINT_ID`). | `src/backend/generation/adapters/cloud/` | **[verified]** |
| **C5** | Register the adapters in the cloud `__init__.py` and `factory.py`; add the endpoint default in `runpod_defaults.py`; wire the family mapping in `app.py` (`"ltx23": "ltx"` currently). | several | **[verified]** |
| **C6** | Decide the LoRA story. 2.5 ships `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (8.9 GB); our 2.3 LoRAs are **not** assumed compatible. Verify against the LoRA registry before exposing LoRA UI for 2.5. | `docs/lora_registry.yaml` | **[needs runtime confirmation]** |

### Phase D — verify (small GPU spend, see §8)

| Step | Action | Status |
|---|---|---|
| **D1** | Deploy a **separate** endpoint for 2.5 (e.g. `oelala-ltx25`) so 2.3 keeps serving. | **[verified as a design choice]** |
| **D2** | Run the single acceptance test in §8. | **[needs runtime confirmation]** |
| **D3** | Only after D2 passes, route traffic and consider retiring the 2.3 worker. | **[needs runtime confirmation]** |

---

## 8. Verifying without wasting GPU money

**One test, one job, one explicit expected outcome.** Everything that can be checked
without a GPU (Phases A-C) is checked first — in particular the whole workflow graph
can be validated for node-class existence and input-name correctness against a local
ComfyUI `/object_info` call, which costs nothing.

### The acceptance test

| Parameter | Value |
|---|---|
| Endpoint | the new 2.5 endpoint only |
| GPU | A40 48 GB (`AMPERE_48`) — cheapest tier that should fit |
| Mode | I2V, then T2V (two jobs, or one if only I2V is needed) |
| Resolution | 960 × 544 (the official graph's default) |
| Frames | **121** (= 1 + 15×8, the official default ≈ 4.84 s @ 25 fps) |
| Steps / sampler | 8, `euler_ancestral`, `cfg = 1`, official `ManualSigmas` |
| Duration cap | worker timeout ≤ 15 min |

**Expected outcome (all must hold):**

1. Cold start logs show all four files downloaded from `Lightricks/LTX-2.5` and
   `ensure_models()` returns no error. **A 403 here means the template is carrying the
   wrong account's token** (§A3 — the most likely failure, since the current `deploy.sh`
   wires `HF_LORA_TOKEN`, which is measured as 403 on this repo). A 401 means no token
   reached the worker at all. **Either way: stop, do not retry blindly.**
2. ComfyUI starts without an import error in `comfy/ldm/lightricks/`.
3. The job produces a playable MP4 with 121 frames at 25 fps.
4. **The output is not noise** — i.e. the model actually loaded rather than falling
   back to random weights. This is the specific failure mode an incompatible
   `comfy/ldm/lightricks/` would produce, and the reason 0.32.0 is required.
5. For I2V, the first frame resembles the input image (confirms
   `LTXVImgToVideoInplace` is wired correctly).
6. If the audio path is enabled, the output has an audio track.

**Cost estimate** (RunPod serverless list prices, page dated 2026-09-27; per second =
per hour ÷ 3600):

| Tier | $/hr | $/s | ~310 s test | ~200 s test |
|---|---:|---:|---:|---:|
| A40 / A6000 48 GB | 1.22 | 0.000339 | **$0.11** | $0.07 |
| L40S / 6000 Ada / MIG 48 GB | 1.75 | 0.000486 | $0.15 | $0.10 |
| A100 80 GB | 2.72 | 0.000756 | $0.23 | $0.15 |
| RTX 6000 Pro 96 GB | 3.49 | 0.000969 | $0.30 | $0.19 |

The ~310 s assumes ~111 s download + ~30 s ComfyUI boot + ~60 s model load + ~90 s
generation + ~20 s teardown. **Only the download time is grounded in verified sizes;
the rest is an estimate.** Budget **$0.25** to be safe. **[needs runtime confirmation]**

**Cheaper pre-check:** a network-volume-backed worker, or a second run on a warm
worker, skips the download and isolates inference cost. Run the *second* job on the
same worker to confirm inference time without paying for the download twice.

---

## 9. Rollback

The migration is designed so that **LTX-2.3 keeps working throughout**.

| Layer | Rollback action | Status |
|---|---|---|
| Registry | `LTX25_MODELS` is a **new** list; `LTX23_MODELS` is untouched. Remove the 2.5 entries to revert. | **[verified: additive design]** |
| Dockerfile | 2.5 changes go in a **new** image tag. Point template `c1fz26l07d` back at the previous dated tag (`20260412-*` or the last known-good). | **[verified]** |
| Endpoint | 2.5 gets its **own** endpoint/template. Delete it and route back to `oelala-ltx23` (`ctpoa610dva4ww`) — the 2.3 worker is never modified in place. | **[verified]** |
| Backend code | `build_cloud_ltx25_*` and `ltx25_*` adapters are **additions**; the 2.3 builders and adapters stay byte-identical. Revert = stop registering the new adapters. | **[verified]** |
| Frontend | If `ltx25` is exposed as a mode, hide it in the model registry; no 2.3 UI change is needed. | **[verified]** |

**Rollback trigger:** any of the §8 expectations 1-4 failing, or the 2.5 worker
exceeding the 2.3 worker's cost per generation by more than ~20 %.

**Important:** the gate acceptance (A1) is **not** reversible and does not need to be —
it grants access to a separate repo and does not affect the 2.3 worker.

---

## 10. Open questions for the operator

1. **NSFW scope (§3.3).** The Lightricks AUP prohibits sexually explicit content and
   the license obliges us to make that restriction enforceable in our own terms. This
   already applies to the deployed 2.3 worker. Does the platform keep serving
   NSFW on LTX at all, or does the second slot become SFW-only? **This is the only
   question that can block the migration on grounds other than engineering.**
2. ~~**Which HF account holds the worker token?**~~ **Answered: `bomehika`.** The gate is
   accepted there and that account has access (206); m0nk111 does not (403). The template
   must carry the `bomehika` token — see §A3, which is a **required fix** before deploying.
3. **GPU tier.** Is a Blackwell tier (`BLACKWELL_96`) in reach? If yes, the 18.7 GB
   nvfp4 transformer is worth evaluating; if no, int8 is the target.
4. **Single-stage or two-stage?** Single-stage is the minimal path (§4.2, 38.7 GB).
   Two-stage adds the 2× spatial upscaler (+1.0 GB) and materially more VRAM/time.
5. **Audio in the second slot — now answered in principle.** LTX-2.5 has **native
   joint audio-video** generation, and the official single-stage graph emits audio
   through `ltx-2.5-audio-vae-bf16` (0.365 GB) + `LTXVAudioVAEDecode`, at no extra
   model cost and **no separate vocoder** (§6.4 item 4). So the second slot can match
   H3's joint audio-video capability. Remaining decision: keep it, given H3 already
   leads on that axis, or drop the AV branch to simplify (W6)? **[verified that the
   capability exists at 0.365 GB; product decision is the operator's]**
6. **LoRA expectations.** Do users expect LTX LoRAs in the second slot? If yes, the
   LoRA compatibility question (C6) becomes a gating item, not a follow-up.
7. **Is `docs/LEGACY.md`'s "code v1.4.x" claim authoritative?** It could not be
   verified against the pack, which has no version at all.

---

## 11. What could not be verified

- **Anything requiring a GPU run:** VRAM fit on a 48 GB A40 with the int8 set; real
  inference time and therefore the exact test cost; whether the int8
  `comfy-int8-convrot` files load through `UNETLoader`/`CLIPLoader` unaided; whether
  the official 2.5 graph we adopt uses `CFGGuider` or `LTXVDualCFGGuider`.
- **That ComfyUI 0.18.1 actually *fails* on 2.5 weights at runtime** — the *cause* is
  now observed (no Gemma-4 text-encoder module and no `gemma4` path in `sd.py` at our
  build commit, §5.1), but the concrete error message has not been seen. No GPU run was
  performed, so this is a code-level certainty rather than a reproduced failure.
- **The 350 MB/s transfer assumption** — taken from the brief, not measured.
- **Token ownership semantics for gated repos** — HF documents gated access per user,
  but I did not test a token belonging to a different account.
- **Whether the official 2.5 two-stage or IC-LoRA workflows add further required
  models** — only the single-stage distilled graph was analysed in depth.
- **"code v1.4.x"** for `ComfyUI-LTXVideo` — no such version string exists in the pack.
- **The community checkpoint's provenance and behaviour** (§4.4) — the REDGraft int8 file
  exists on the public mirror at 17.026 GB, but its quant format, tensor layout,
  `UNETLoader` compatibility and output quality were not validated, and the Civitai
  model/version/file IDs it is attributed to were not independently checked.
- **The exact ComfyUI-LTXVideo and VHS/KJNodes commits** baked into the currently
  deployed image — the Dockerfile records no pins and the image tag only gives a date.
- **Whether LTX-2.5's `Lightricks/LTX-2.5` repo has a `LICENSE` file** — it does not
  (the file listing shows only `.gitattributes`, `README.md`, `hf-hero-web.webp` and
  weights). Note also that the model card's licence link is **broken**: it points at
  `Lightricks/LTX-2/blob/main/LICENSE.md`, which returns **404**. The real file is
  `LICENSE` — a 537-byte pointer that names both agreements and their dates:
  *LTX-2.x Community License Agreement*, **dated August 11, 2026** (applies to 2.5 and
  all future 2.x); *LTX-2 Community License Agreement*, **dated January 5, 2026**
  (applies to 2.3 until August 11, 2026). **[verified]**

---

## 12. Sources

**HuggingFace (metadata + license)**
- `https://huggingface.co/api/models/Lightricks/LTX-2.5?blobs=true` — gating, file listing, sizes
- `https://huggingface.co/api/models/Lightricks/LTX-2.3?blobs=true` — comparison
- `https://huggingface.co/api/models/Comfy-Org/ltx-2?blobs=true` — current 2.3 text encoder
- `https://huggingface.co/api/models/Comfy-Org/gemma-4?blobs=true` — Gemma 4 encoders
- `https://huggingface.co/api/models/Comfy-Org/ltx-2.3?blobs=true` — 2.3-era repack (ungated)
- `https://huggingface.co/api/models/bomehika/oelala-models?blobs=true` — public mirror (ungated), community LTX-2.5 int8 checkpoint
- `https://huggingface.co/Lightricks/LTX-2.5` — gating banner

**License**
- `https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2_x` — LTX-2.x Community License (2.5), 30 399 B, dated 2026-08-11
- `https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2` — LTX-2 Community License (2.3), 19 197 B, dated 2026-01-05
- `https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE` — 537 B pointer naming both agreements and their dates
- `https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf` — Acceptable Use Policy (5 pages, revision dated 2026-03-30)
- `https://huggingface.co/docs/hub/en/models-gated` — `auto` vs manual gate approval semantics

**ComfyUI core**
- `https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.32.0` — "Add support for LTX 2.5" (PR #15499), min torch 2.7
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.31.0/comfy_extras/nodes_lt.py` — 0 × `LTXVDualCFGGuider` (below the floor)
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.32.0/comfy_extras/nodes_lt.py` — 3 × `LTXVDualCFGGuider` (at the floor)
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.31.0/comfy/text_encoders/gemma4.py` — no `gemma4_text_encoder_model`
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c/comfyui_version.py` — our build = 0.18.1
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c/comfy/text_encoders/gemma4.py` — **404** at our build (2.5 TE path absent)
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c/comfy/sd.py` — 0 × `gemma4` (v0.32.0: 21)
- `https://github.com/Comfy-Org/ComfyUI/pull/15499` — "Add support for LTX 2.5", merged 2026-08-11
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.38.0/folder_paths.py` — model folder names
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.38.0/comfy/model_management.py` — `supports_nvfp4_compute` (CC ≥ 10) vs `supports_int8_compute` (no CC gate)
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.38.0/comfy/ops.py` — `get_disabled_quant_formats`, `int8_tensorwise` + `convrot`
- `https://github.com/Comfy-Org/ComfyUI/commit/54e03f5367ebd8d96380e4cf02fa3084f7a7eca5` — 2.5 blueprint added (0.35.0)
- `https://github.com/Comfy-Org/ComfyUI/commit/830232b856045ca2892833212d7771078a13edd5` — v0.37.2 tag, 2026-09-23

**ComfyUI-LTXVideo**
- `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/README.md`
- `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/__init__.py` — node class registry
- `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/requirements.txt`
- `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json`
- `https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/2.5/README.md`

**ComfyUI official workflow templates** (separate repo — this is why a search inside the
ComfyUI repo for these files returns 404)
- `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_t2v.json` — built-in 2.5 T2V template; source for the int8 default set, `LTXVDualCFGGuider` and the stage-2 `0.85` sigmas

**Pricing**
- `https://www.runpod.io/pricing` — serverless per-hour rates, page dated 2026-09-27

**Repository files**
- `deploy/runpod-ltx23/Dockerfile`, `handler.py`, `deploy.sh`
- `src/backend/comfyui_client.py`, `src/backend/generation/adapters/cloud/ltx23_{t2v,i2v}.py`
- `docs/LEGACY.md`, `docs/RUNPOD_GPU_TIERS.md`
