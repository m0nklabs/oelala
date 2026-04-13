# LoRA Usage Documentation

> Auto-researched from CivitAI on 2026-04-13 via SHA256 hash lookup.

---

## LTX 2.3 LoRAs (8)

### 1. DR34ML4Y_LTXXX_PREVIEW_RC1

| Field | Value |
|-------|-------|
| **Status** | ❌ NOT FOUND on CivitAI |
| **Local file** | `DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors` |
| **Base model** | LTXV 2.3 |
| **Trigger words** | Unknown |
| **Strength** | Unknown |
| **Notes** | Community/private release. Not available on CivitAI. |

---

### 2. LTX-2.3 - Ahegao Face v1

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2138899 |
| **Model name** | Ahegao (T2V & I2V) |
| **Creator** | playtime_ai_ |
| **Version** | LTX-2.3 |
| **Base model** | LTXV 2.3 |
| **Local file** | `LTX-2.3 - Ahegao Face v1.safetensors` |
| **Trigger words** | None (use natural language, e.g. "She makes the ahegao face by sticking her tongue out and crossing her eyes.") |
| **Strength** | Default (1.0) |
| **Notes** | Multi-platform model — also has Wan 2.2 T2V, LTX-2, Pony, Illustrious, SDXL, Flux Klein, Z Image versions. |

---

### 3. SexGod_Nudity_LTX23_v2_0

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2308157 |
| **Model name** | SexGod LTX 2.3 Female Nudity |
| **Creator** | sexgod1979 |
| **Version** | v2.0 |
| **Base model** | LTXV 2.3 |
| **Local file** | `SexGod_Nudity_LTX23_v2_0.safetensors` |
| **Trigger words** | `LTXNUDES` |
| **Strength** | 1.0 (with Kijai distil lora at 0.60) |
| **Notes** | Trained on 105 high-res nude female videos (I2V and T2V). Best with strong starting image (I2V). ~417 frames recommended (10-15s videos). Trained at 768+ pixel buckets. Use undistilled base model (GGUF bf16 or Q_8) with detailer lora at 1.0. Natural language prompts. |

---

### 4. bounceV2_LTX23_I2V

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1343431 |
| **Model name** | Bouncing Boobs - LTX / Wan |
| **Creator** | ai_build_art |
| **Version** | LTX 2.3 |
| **Base model** | LTXV 2.3 |
| **Local file** | `bounceV2_LTX23_I2V.comfy.safetensors` |
| **Trigger words** | `her breast is bouncing up and down`, `her breast is bouncing from left to right` |
| **Strength** | Default (1.0) |
| **Notes** | First LTX 2.3 version. Also has Wan 2.2 I2V high/low noise versions. Creator notes "room for improvement" on LTX 2.3. |

---

### 5. head_swap_v3

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2027766 |
| **Model name** | BFS - Best Face Swap |
| **Creator** | NRDX |
| **Version** | LTX-2.3 - V3 Focus Head |
| **Base model** | LTXV 2.3 |
| **Local file** | `head_swap_v3_rank_adaptive_fro_098.safetensors` |
| **Trigger words** | `head_swap: FACE: [describe the new face here] ACTION: [describe the action from original video here]` |
| **Strength** | Default (1.0) |
| **Notes** | V3 uses persistent-template conditioning — keeps new face visible throughout entire guide sequence (unlike V1/V2 which relied on Frame 0 only). Also has Flux Klein 9B/4B, LTX-2, and Qwen versions. No Wan 2.2 version exists. |

---

### 6. ltx2.3_nsfw_furry

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2310920 |
| **Model name** | LTX 2/2.3 [I2V] NSFW (+furry) - Multi purpose sex lora |
| **Creator** | mylo1337 |
| **Version** | v2.0 LTX2.3 step 36000 |
| **Base model** | LTXV 2.3 |
| **Local file** | `ltx2.3_nsfw_furry.safetensors` |
| **Trigger words** | None |
| **Strength** | Varies — see description: may desaturate at full strength on fp8. Use gguf version if desaturation occurs. |
| **Notes** | Retrained from scratch for LTX 2.3, 5x lower learning rate, 2 gradient accumulation. Also has merge versions (v1 x0.5/v2 x0.7 and v1 x0.4/v2 x1.0). LTX-only, no Wan version from this creator. |

---

### 7. ltxdeepthroat_v01

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2476698 |
| **Model name** | LTX-2.3 Deepthroat |
| **Creator** | daring_l |
| **Version** | v0.1 |
| **Base model** | LTXV 2.3 |
| **Local file** | `ltxdeepthroat_v01.safetensors` |
| **Trigger words** | `LTXdeepthroat` |
| **Strength** | Stage 1: 1.0, Stage 2: varies (check workflow) |
| **Notes** | Early release (v0.1). Rank 64 (~1GB, large). Audio glottal sounds need more work. Example workflows included in training data zip. Same creator as sfbehind LTX. |

---

### 8. sfbehind_LTX2_3

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2298764 |
| **Model name** | Sex from behind (Facing Cam) |
| **Creator** | daring_l |
| **Version** | LTX-2.3 v1 |
| **Base model** | LTXV 2.3 |
| **Local file** | `sfbehind_LTX2_3_v0_1.safetensors` |
| **Trigger words** | None |
| **Strength** | Default (1.0) |
| **Notes** | Positions: doggy style, prone, top-down bottom-up — all facing camera. Also has LTX-2 versions (v0.1, v0.2). Image captioning system included. Same creator has separate Wan I2V version (see #17). |

---

## Wan 2.2 LoRAs (11)

### 9. wan_ahegao_v2

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2138899 |
| **Model name** | Ahegao (T2V & I2V) |
| **Creator** | playtime_ai_ |
| **Version** | Wan2.2 - T2V v4 - HIGH/LOW |
| **Base model** | Wan Video 2.2 T2V-A14B |
| **Local file(s)** | `Wan2.2 - T2V - Ahegao v4 - HIGH 14B.safetensors` / `LOW` variant |
| **Trigger words** | `She makes the ahegao face. She sticks her tongue out and crosses her eyes.` |
| **Strength** | Default (1.0) |
| **Notes** | Same parent model as LTX Ahegao (#2). T2V versions only for Wan 2.2. Also has Wan 2.1 I2V version. HIGH = high noise, LOW = low noise variants. |

---

### 10. wan_bounceV2_i2v_trainer_v2

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1343431 |
| **Model name** | Bouncing Boobs - LTX / Wan |
| **Creator** | ai_build_art |
| **Version** | WAN 2_2 Bounce High / Low |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `BounceHighWan2_2.safetensors` / `BounceLowWan2_2.safetensors` |
| **Trigger words** | `her breasts are bouncing` |
| **Strength** | Default (1.0) |
| **Notes** | Same parent model as LTX bounce (#4). I2V versions for Wan 2.2. Use both HIGH and LOW noise variants. |

---

### 11. wan_deepthroat_v01

> **3 separate deepthroat models exist on CivitAI:**

#### 11a. jfj-deepthroat (by JeeFJ)

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1497390 |
| **Model name** | Deepthroat, Blowjob - Wan 2.X I2V & T2V |
| **Creator** | JeeFJ |
| **Base model** | Wan Video 2.2 T2V-A14B (T2V) / I2V-A14B (I2V) |
| **Local file(s)** | `jfj-deepthroat-W22-T2V-HN-v1.safetensors` / `LN-v1` / `I2V-HN` / `I2V-LN` |
| **Trigger words** | `blowjob, deepthroat` |

#### 11b. Wan22_ThroatV3 (by Civitaiwolfxx)

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2023407 |
| **Model name** | Deepthroat/Face Fuck - Wan2.2 I2V |
| **Creator** | Civitaiwolfxx |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `Wan22_ThroatV3_High.safetensors` / `Low` / `Low_Alt` |
| **Trigger words** | Long scenario prompt (see face fuck section #12) |

#### 11c. Ultimate DeepThroat (by K3NK)

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1874811 |
| **Model name** | Ultimate DeepThroat I2V Wan2.2 Video LoRa - K3NK |
| **Creator** | K3NK |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `wan22-ultimatedeepthroat-i2v-102epoc-high-k3nk.safetensors` / `low` |
| **Trigger words** | `a woman in front of a penis, she engages in a deep throat blowjob, she swallows the penis all the way. Her lips touches the man's groin. Her nose smashes against the man's hips.` |

---

### 12. wan_facefuck_v01

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2023407 |
| **Model name** | Deepthroat/Face Fuck - Wan2.2 I2V |
| **Creator** | Civitaiwolfxx |
| **Version** | v3.0 (HIGH/LOW) |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `Wan22_ThroatV3_High.safetensors` / `Wan22_ThroatV3_Low.safetensors` |
| **Trigger words** | `a woman is kneeling in front of a man, she is giving him a deepthroat blowjob, almost the entire penis is in her mouth, he is grabbing her head and holding it in place, then he sharply pulls her head back until the penis is no longer in her mouth, she is gasping for air, then he quickly grabs her head and pulls her head towards his penis, she starts giving him a deepthroat blowjob, the view is from the side` |
| **Strength** | Default (1.0) |
| **Notes** | Combined deepthroat + face fuck model. Has v1-v3 versions. Face fuck triggers in v2: "he starts face fucking her, he thrusts his hips back and forth". Same model as deepthroat #11b. |

---

### 13. wan_headswap_v4

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2027766 |
| **Model name** | BFS - Best Face Swap |
| **Creator** | NRDX |
| **Version** | QIE 2509 - V4 Focus Head |
| **Base model** | ⚠️ **Qwen Image Edit** (NOT Wan 2.2!) |
| **Local file** | `bfs_head_swap_v4.safetensors` |
| **Trigger words** | `h34d_sw4p: replace the head of Picture 1 by the head from Picture 2, strictly preserving the identity, facial features (eyes, nose, mouth), and skin texture of Picture 2. Ensure the new head mimics the identical expression, angle, and rotation found in Picture 1.` |
| **Strength** | Default (1.0) |
| **Notes** | ⚠️ This is a Qwen Image Edit LoRA, NOT a Wan 2.2 video LoRA. No Wan 2.2-specific head swap version exists from BFS. Use the LTX-2.3 V3 version for video head swapping instead. |

---

### 14. wan_missionary_v02

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1331682 |
| **Model name** | Wan 2.2/2.1 POV Missionary |
| **Creator** | dtwr434 |
| **Version** | Wan2.2 I2V Highnoise / Lownoise |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors` / `lownoise` |
| **Trigger words** | None |
| **Strength** | Default (1.0) |
| **Notes** | Also has T2V high/low noise versions. Workflows included in download. Older Wan 2.1 versions also available. |

---

### 15. wan_nsfw_furry_v1

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1782485 |
| **Model name** | Furry Enhancer Video |
| **Creator** | freek22 |
| **Version** | WAN 2.2 I2V V3.0 HighNoise / LowNoise |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `Furry Enhancer Wan2.2 V3 High Noise I2V.safetensors` / `Low Noise` |
| **Trigger words** | `anthro`, `furry` (V2 triggers; V3 has no explicit triggers) |
| **Strength** | Default (1.0) |
| **Notes** | Multi-platform: also has LTX 2.3, Wan 2.2 T2V, Wan 2.1, and Hunyuan versions. V3 is latest. Workflows included. Different model from LTX nsfw_furry (#6) which is by mylo1337. |

---

### 16. wan_reverse_cowgirl_v01

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1428098 |
| **Model name** | WAN COWGIRL + REVERSE COWGIRL -- T2V & I2V LoRa |
| **Creator** | ERA5ER |
| **Version** | wan2.2-I2V-14B-HIGH/LOW v1 |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `wan22.r3v3rs3_c0wg1rl-14b-High-i2v_e70.safetensors` / `Low` |
| **Trigger words** | `r3v3rs3_c0wg1rl`, `c0wg1rl`, `straddling him in the reverse cowgirl position` |
| **Strength** | Default (1.0) |
| **Notes** | Combined cowgirl + reverse cowgirl model. Also has T2V high/low, 5B TI2V, and Wan 2.1 versions. |

---

### 17. wan_sfbehind_v6

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2227622 |
| **Model name** | I2V - Sex from behind (front facing) |
| **Creator** | daring_l |
| **Version** | v2.1 high_noise / low_noise |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `sfbehind_v2.1_high_noise.safetensors` / `low_noise` |
| **Trigger words** | `sfb3hind`, `sfbehind` |
| **Strength** | Default (1.0) |
| **Notes** | Separate model from LTX sfbehind (#8) — same creator (daring_l) but different CivitAI model IDs. I2V-only. Has v1 and v2.1 versions. |

---

### 18. wan_titfuck_v02

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/2070335 |
| **Model name** | WAN 2.2 I2V - POV Paizuri / Titfuck |
| **Creator** | TwoMoreLurker |
| **Version** | HIGH v1.0 / LOW v1.0 |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `WAN-2.2-I2V-POV-Titfuck-Paizuri-HIGH-v1.0.safetensors` / `LOW` |
| **Trigger words** | `titJob`, `girlMove`, `manMove`, `gather`, `fingersTogether` |
| **Strength** | Default (1.0) |
| **Notes** | I2V-only. Two noise variants (HIGH/LOW). |

---

### 19. wanride_v02

| Field | Value |
|-------|-------|
| **CivitAI URL** | https://civitai.com/models/1452829 |
| **Model name** | BBC Ride Wan (2.2!) |
| **Creator** | dngstn32 |
| **Version** | Wan 2.2 I2V High / Low |
| **Base model** | Wan Video 2.2 I2V-A14B |
| **Local file(s)** | `bbcRide_wan22_I2V_high_e30.safetensors` / `low_e30` |
| **Trigger words** | None |
| **Strength** | Default (1.0) |
| **Notes** | Also has T2V high/low versions. Re-trained from Wan 2.1 for Wan 2.2. Use BOTH high and low LoRAs together. |

---

## Summary Table

| # | Name | CivitAI ID | Creator | Triggers | Found? |
|---|------|------------|---------|----------|--------|
| 1 | DR34ML4Y_LTXXX_PREVIEW_RC1 | — | — | Unknown | ❌ |
| 2 | Ahegao Face v1 (LTX) | 2138899 | playtime_ai_ | Natural language | ✅ |
| 3 | SexGod Nudity (LTX) | 2308157 | sexgod1979 | `LTXNUDES` | ✅ |
| 4 | bounceV2 (LTX) | 1343431 | ai_build_art | "her breast is bouncing..." | ✅ |
| 5 | head_swap v3 (LTX) | 2027766 | NRDX | `head_swap: FACE:...` | ✅ |
| 6 | nsfw_furry (LTX) | 2310920 | mylo1337 | None | ✅ |
| 7 | deepthroat (LTX) | 2476698 | daring_l | `LTXdeepthroat` | ✅ |
| 8 | sfbehind (LTX) | 2298764 | daring_l | None | ✅ |
| 9 | ahegao (Wan) | 2138899 | playtime_ai_ | "She makes the ahegao face..." | ✅ |
| 10 | bounce (Wan) | 1343431 | ai_build_art | "her breasts are bouncing" | ✅ |
| 11 | deepthroat (Wan) | Multiple | Multiple | Multiple | ✅ |
| 12 | facefuck (Wan) | 2023407 | Civitaiwolfxx | Long scenario | ✅ |
| 13 | headswap v4 (Wan) | 2027766 | NRDX | `h34d_sw4p:...` | ⚠️ Qwen only |
| 14 | missionary (Wan) | 1331682 | dtwr434 | None | ✅ |
| 15 | nsfw_furry (Wan) | 1782485 | freek22 | `anthro`, `furry` | ✅ |
| 16 | reverse_cowgirl (Wan) | 1428098 | ERA5ER | `r3v3rs3_c0wg1rl` | ✅ |
| 17 | sfbehind (Wan) | 2227622 | daring_l | `sfb3hind`, `sfbehind` | ✅ |
| 18 | titfuck (Wan) | 2070335 | TwoMoreLurker | `titJob`, `girlMove`, etc. | ✅ |
| 19 | wanride (Wan) | 1452829 | dngstn32 | None | ✅ |

---

## Warnings

1. **wan_headswap_v4** (`bfs_head_swap_v4.safetensors`) is for **Qwen Image Edit**, not Wan 2.2 video. No Wan 2.2 video head swap LoRA exists from BFS.
2. **DR34ML4Y_LTXXX_PREVIEW_RC1** is not on CivitAI — likely a private/community release.
3. **wan_deepthroat** has **3 different models** from 3 creators. Choose based on use case (T2V vs I2V, mouth-focus vs scenario-based).
4. **LTX nsfw_furry** (#6, mylo1337) and **Wan nsfw_furry** (#15, freek22) are from **different creators** and are separate models.
5. **sfbehind LTX** (#8) and **sfbehind Wan** (#17) are by the **same creator** (daring_l) but different CivitAI model IDs.
