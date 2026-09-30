### Added
- Prompt-generator LLM options:
  - **Model choice for the H3 skill** — the pinned Huihui-Qwen3.5-9B is now just the
    default; any model in the list can be picked for H3 targets.
  - **Scene mode** on `/generate-prompt` (`scene_mode`): `random` (dice roll, the
    previous behaviour), `creative` (the model picks place, lighting, wardrobe, mood,
    style and story wrapper from the vocabulary first, then writes the shots around
    it — with a dice-roll fallback when the picker answer is unusable) or off.
  - **Per-call controls** — `temperature` and `llm_seed` are now request fields
    (defaults unchanged: 1.2 / fresh seed), plus a temperature slider and reset in the
    Prompt Generator.
  - **Model comparison** — `POST /generate-prompt/compare` runs the same request
    through up to three models; jobs go through the LLM queue and the client polls
    `/llm-job/{id}`, so a slow H3 skill run cannot time the request out. The Prompt
    Generator shows the results side by side with copy buttons.
