/**
 * Shared LLM model definitions for all tools.
 * Single source of truth — import from here, never hardcode model lists inline.
 *
 * To add/remove a model: edit ONLY this file, all tools update automatically.
 */

// ── Text LLMs (prompt enhancement / generation) ────────────────────────────
export const PROMPT_LLM_MODELS = [
  { id: 'GLM-4.7-Flash-Claude-Opus-Reasoning', label: 'GLM+Claude ✨', labelLong: 'GLM-4.7 + Claude Opus', description: 'Default · Claude Opus reasoning distillation · best quality' },
  { id: 'GLM-4.7-Flash', label: 'GLM Flash', labelLong: 'GLM-4.7 Flash', description: 'Fast · compact · good for simple prompts' },
  { id: 'GLM-4.7-Flash-Uncensored-Balanced', label: 'GLM Uncensored', labelLong: 'GLM-4.7 Uncensored', description: 'No content filters · balanced output' },
  { id: 'Qwen3-30B-A3B-Thinking-2507', label: 'Qwen3 30B', labelLong: 'Qwen3 30B Thinking', description: 'High quality · reasoning mode · slower' },
  { id: 'gemma-3-27b-it', label: 'Gemma 27B', labelLong: 'Gemma 3 27B', description: 'Google · strong creative writing' },
]

export const DEFAULT_PROMPT_LLM = 'GLM-4.7-Flash-Claude-Opus-Reasoning'

// ── NSFW / Uncensored text LLMs (no content filters) ───────────────────────
export const NSFW_LLM_MODELS = [
  { id: 'GLM-4.7-Flash-Uncensored-Balanced', label: 'GLM Uncensored 🔥', labelLong: 'GLM-4.7 Uncensored', description: 'Best NSFW · no content filters · uncensored' },
  { id: 'DeepSeek-R1-Distill-Qwen-32B-Uncensored', label: 'DeepSeek 32B', labelLong: 'DeepSeek R1 32B Uncensored', description: 'Large · reasoning · uncensored · slow' },
  { id: 'gpt-oss-20b-uncensored', label: 'GPT-OSS 20B', labelLong: 'GPT-OSS 20B Uncensored', description: 'Uncensored · good creative writing' },
  { id: 'GLM-4.7-Flash-Claude-Opus-Reasoning', label: 'GLM+Claude ✨', labelLong: 'GLM-4.7 + Claude Opus', description: 'Best quality · mostly uncensored' },
]

export const DEFAULT_NSFW_LLM = 'GLM-4.7-Flash-Uncensored-Balanced'

// ── Vision LLMs (image/video captioning) ────────────────────────────────────
export const VISION_MODELS = [
  { id: 'Huihui-gemma-4-26B-A4B-it-abliterated', label: 'Gemma 4 26B MoE', description: 'Best quality · abliterated · fast' },
  { id: 'gemma-4-31B-it-uncensored-heretic', label: 'Gemma 4 31B Heretic', description: 'Full 31B · uncensored · slower' },
  { id: 'Gemma3-27B-it-vl-GLM-4.7-Uncensored-Heretic-Deep-Reasoning', label: 'Gemma3 27B VL Heretic', description: 'Vision + deep reasoning · uncensored' },
  { id: 'Step3-VL-10B', label: 'Step3-VL 10B', description: 'Fast · good quality' },
]

export const DEFAULT_VISION_MODEL = 'Huihui-gemma-4-26B-A4B-it-abliterated'
