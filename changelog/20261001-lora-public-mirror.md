### Added

- Public LoRA delivery mirror `bomehika/oelala-loras` (126 files, 52.7 GB) with an
  adult-content notice in its model card. Cloud workers download from it
  anonymously, so worker templates no longer need a HuggingFace token for LoRAs.
- `scripts/lora_public_exclusions.txt`: the policy
  list of LoRAs that must never reach a public mirror (face-swap capability or a
  real person's likeness), covered by
  `tests/test_lora_public_exclusions.py`.
- `LORA_HF_MIRROR_TOKEN`: token sent for `LORA_HF_MIRROR_REPO`. Empty means
  anonymous (public mirror); unset falls back to `HF_LORA_TOKEN`.
- `scripts/build_lora_source_index.py` + `data/lora_source_index.json`: per-file
  knowledge of which third-party dump holds which LoRA, so the primary URL is
  chosen instead of firing a 404 first. Six public dumps are indexed
  (`Serenak/chilloutmix`, `jaysowen/wan2.2-nsfw-loras`,
  `kingofpersia/wan-22-nsfw-loras`, `lkzd7/WAN2.2_LoraSet_NSFW`,
  `JERRYNPC/WAN2.2-LORA-NSFW`, `diobrando0/LTX_NSFW_loras`), covering 33 of the
  128 local LoRAs; the remaining 95 are served by our own mirror. The builder only
  accepts files whose byte size matches the local store (revision guard) and
  records the in-repo path, so dumps that keep files in subdirectories resolve.
- `generation/lora_public_policy.py`: single source of truth for the exclusion
  list, used by both the uploader and the download path.
- `scripts/sync_loras_hf.py` flags: `--allow-public`, `--token-env`,
  `--exclusions`, `--only-excluded`.

### Changed

- Subdir-mirror URLs percent-encode the path, so LoRAs with spaces or
  subdirectories (e.g. `wan 2.2/…`) resolve; the unencoded form failed outright.
- The six policy-excluded LoRAs were mirrored to the private repo
  `m0nk111/oelala-loras` instead of the public one.
- Download priority stays third-party-first (`Serenak/chilloutmix` dump before our
  own mirrors): the download traffic should land on someone else's account. The
  docstrings and tests now state that intent explicitly instead of describing the
  order as an accident.
- `build_lora_download_list()` is now exclusion-aware: a file listed in
  `scripts/lora_public_exclusions.txt` skips every public source and goes straight
  to the signed self-hosted URL, so face-swap / real-person files never get a
  public download URL at all.

### Removed

- RunPod LoRA Network Volume `ochebt0xbq` (`oelala-runpod-lora-eu-cz`, `EU-CZ-1`,
  50 GB, ~$3.50/month): all 91 objects were byte-identical to files in the local
  store and no endpoint had it attached (`networkVolumeId` was null on all three
  serverless endpoints). `RUNPOD_S3_*`, `RUNPOD_LORA_VOLUME_*` and
  `RUNPOD_LORA_S3_ENDPOINT` are gone from `.env.example`, and the
  `.github/copilot-instructions.md` storage policy no longer describes a LoRA
  volume.
