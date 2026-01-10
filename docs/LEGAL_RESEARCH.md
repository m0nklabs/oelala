# Legal Research: Content Policy & Licensing

> Status: **RESEARCH NEEDED**
> Priority: HIGH - Must resolve before commercial launch

---

## 1. Content Policy for Oelala

### Question: Do we have content rules similar to LTX-2's restrictions?

**LTX-2 Restrictions (from license):**
- ❌ Exploiting/harming minors
- ❌ Deepfakes without consent
- ❌ Harassment/defamation
- ❌ Discrimination
- ❌ Military/weapons
- ❌ Malware
- ❌ False information/disinformation

### Current Oelala Status

**What we HAVE:**
- [x] NSFW toggle (forced off for guests)
- [x] Age gate (login = 18+)
- [x] SFW/NSFW tagging system
- [x] Gallery filtering for anonymous users

**What we DON'T HAVE:**
- [ ] Written Terms of Service
- [ ] Content Policy / Acceptable Use Policy
- [ ] DMCA takedown procedure
- [ ] User reporting system
- [ ] Moderation guidelines
- [ ] Privacy Policy
- [ ] Cookie Policy

### TODO: Create Content Policy

```markdown
Minimum content policy should cover:

1. PROHIBITED CONTENT (absolute bans):
   - CSAM / minors in any sexual context
   - Real person deepfakes without consent
   - Revenge porn / non-consensual intimate images
   - Content promoting violence/terrorism
   - Hate speech / discrimination
   - Malware / illegal activities

2. RESTRICTED CONTENT (requires login + NSFW tag):
   - Adult/explicit content
   - Gore/violence (non-photorealistic)
   - Controversial themes

3. MODERATION:
   - User reporting mechanism
   - Takedown procedures
   - Appeal process

4. LEGAL COMPLIANCE:
   - DMCA procedure
   - 18 U.S.C. §2257 compliance (if hosting explicit)
   - GDPR (if EU users)
```

---

## 2. CivitAI LoRA Licensing

### Question: Can we use CivitAI-downloaded LoRAs in Oelala?

### CivitAI License System

From CivitAI ToS Section 9.4:
> "You may choose to make your Models available for download or use by other Users. In such circumstances, you may license your Models under **bespoke license terms** ("Bespoke License") generated via the Service."

This means: **EACH MODEL HAS ITS OWN LICENSE!**

### Common CivitAI License Types

| License | Commercial Use | Redistribution | Modification |
|---------|---------------|----------------|--------------|
| **CreativeML Open RAIL-M** | ✅ Yes | ✅ Yes | ✅ Yes |
| **CreativeML Open RAIL++-M** | ✅ Yes | ✅ Yes | ✅ Yes |
| **SDXL License** | ✅ Yes* | ✅ Yes | ✅ Yes |
| **Proprietary / No Commercial** | ❌ No | ❌/varies | varies |
| **CC-BY-NC** | ❌ No | ✅ Yes | ✅ Yes |
| **CC-BY-SA** | ✅ Yes | ✅ Yes (same license) | ✅ Yes |

*SDXL License has restrictions on competing with Stability AI

### Our LoRAs - Need to Check

**Location:** `/home/flip/oelala/ComfyUI/models/loras/`

**TODO: For each LoRA we use:**
1. Check the license on CivitAI page
2. Document in a LICENSE_AUDIT.md
3. Remove or replace any non-commercial LoRAs

### Key Questions

1. **Are we using LoRAs commercially?**
   - If Oelala is free → probably OK for most licenses
   - If Oelala has paid features → need commercial licenses only

2. **Are we redistributing the LoRAs?**
   - If users download the LoRA files → redistribution
   - If we only serve generated images → probably not redistribution

3. **Are we modifying LoRAs?**
   - Probably not directly, but generated images are derivatives

### Safe Approach

```
1. Only use LoRAs with CreativeML Open RAIL-M or similar permissive licenses
2. Document all LoRAs and their licenses
3. Credit creators where required
4. Avoid "No Commercial Use" LoRAs entirely
5. Consider creating our own LoRAs for critical use cases
```

---

## 3. Base Model Licensing

### Models We Use

| Model | License | Commercial | Notes |
|-------|---------|------------|-------|
| **WAN 2.2 14B** | ? | ? | Need to check |
| **LTX-2** | LTX-2 Community License | ✅ <$10M revenue | Free for <$10M annual revenue |
| **SDXL** | SDXL License | ✅ Yes* | Can't compete with Stability |
| **Flux** | ? | ? | Need to check |

### TODO: Check All Base Models

---

## 4. Action Items

### Immediate (Before Launch)
- [ ] Create Terms of Service document
- [ ] Create Privacy Policy
- [ ] Create Content Policy / AUP
- [ ] Audit all LoRAs for licenses
- [ ] Audit all base models for licenses
- [ ] Add 18+ age verification modal
- [ ] Add content reporting mechanism

### Short-term
- [ ] DMCA takedown procedure
- [ ] Cookie consent banner (GDPR)
- [ ] User agreement checkbox on signup
- [ ] Moderation queue/tools

### Long-term
- [ ] Automated content moderation (NSFW detection)
- [ ] Appeal process
- [ ] Creator terms (if users upload)

---

## 5. Reference Links

- [CivitAI ToS](https://civitai.com/content/tos)
- [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- [LTX-2 License](https://huggingface.co/Lightricks/LTX-2/blob/main/LICENSE)
- [SDXL License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)

---

## 6. Quick Answers for Flip

### "Hebben wij al dit soort regels op oelala?"
**Nee, we hebben alleen technische NSFW filtering, geen geschreven policies.**

### "Mag ik CivitAI LoRAs zomaar gebruiken?"
**Hangt af van de individuele LoRA licentie. Elke LoRA op CivitAI heeft zijn eigen licentie. Je moet per LoRA checken.**

Safe bet: Gebruik alleen LoRAs met "CreativeML Open RAIL-M" of vergelijkbare permissive licenties.

---

*Last updated: 2026-01-08*
