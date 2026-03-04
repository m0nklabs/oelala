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
  { id: 'GLM-4.7-Flash-Uncensored-Aggressive', label: 'GLM Aggressive 🔥', labelLong: 'GLM-4.7 Uncensored Aggressive', description: 'Maximum uncensored · no filters · best for explicit' },
  { id: 'GLM-4.7-Flash-Uncensored-Balanced', label: 'GLM Uncensored', labelLong: 'GLM-4.7 Uncensored Balanced', description: 'Uncensored · balanced output' },
  { id: 'DeepSeek-R1-Distill-Qwen-32B-Uncensored', label: 'DeepSeek 32B', labelLong: 'DeepSeek R1 32B Uncensored', description: 'Large · reasoning · uncensored · slow' },
  { id: 'gpt-oss-20b-uncensored', label: 'GPT-OSS 20B', labelLong: 'GPT-OSS 20B Uncensored', description: 'Uncensored · good creative writing' },
  { id: 'GLM-4.7-Flash-Claude-Opus-Reasoning', label: 'GLM+Claude ✨', labelLong: 'GLM-4.7 + Claude Opus', description: 'Best quality · mostly uncensored' },
]

export const DEFAULT_NSFW_LLM = 'GLM-4.7-Flash-Uncensored-Aggressive'

// ── Vision LLMs (image/video captioning) ────────────────────────────────────
export const VISION_MODELS = [
  { id: 'Qwen3-VL-32B-Gemini-Heretic-Uncensored-Thinking', label: 'Qwen3-VL 32B Heretic', description: 'Best quality · uncensored · slow' },
  { id: 'Gemma3-27B-it-vl-GLM-4.7-Uncensored-Heretic', label: 'Gemma3 27B VL Heretic', description: 'Vision + reasoning · uncensored' },
  { id: 'Qwen3-VL-30B-A3B-Thinking', label: 'Qwen3-VL 30B MoE', description: 'MoE · thinking mode · fast' },
  { id: 'Step3-VL-10B', label: 'Step3-VL 10B', description: 'Fast · good quality' },
  { id: 'moondream', label: 'Moondream', description: 'Ultra-light · fastest' },
]

export const DEFAULT_VISION_MODEL = 'Qwen3-VL-32B-Gemini-Heretic-Uncensored-Thinking'
