/**
 * Unit tests for parseComfyWorkflow utility.
 * Run with:  node tests/test_parseComfyMetadata.mjs
 */

import { parseComfyWorkflow } from '../src/frontend/src/utils/parseComfyMetadata.js'

let passed = 0, failed = 0

function assert(label, got, expected) {
  if (JSON.stringify(got) === JSON.stringify(expected)) {
    console.log(`  ✅ ${label}`)
    passed++
  } else {
    console.error(`  ❌ ${label}  →  expected ${JSON.stringify(expected)}, got ${JSON.stringify(got)}`)
    failed++
  }
}

// ─── Test 1: Wan2.2 I2V-style workflow with _meta.title ──────────────────────
console.log('\n=== Test 1: Wan2.2 I2V with _meta.title ===')
{
  const workflow = {
    "7":  { class_type: "CLIPTextEncode", inputs: { text: "a cat on the moon" }, _meta: { title: "Positive Prompt" } },
    "8":  { class_type: "CLIPTextEncode", inputs: { text: "blurry, ugly" }, _meta: { title: "Negative Prompt" } },
    "10": { class_type: "KSamplerAdvanced", inputs: { add_noise: "enable",  positive: ["7",0], negative: ["8",0], steps: 6, cfg: 3.0, sampler_name: "uni_pc", scheduler: "karras", noise_seed: 42 } },
    "11": { class_type: "KSamplerAdvanced", inputs: { add_noise: "disable", positive: ["7",0], negative: ["8",0], steps: 6, cfg: 3.0, sampler_name: "uni_pc", scheduler: "karras", noise_seed: 42 } },
    "16": { class_type: "WanImageToVideo", inputs: { width: 480, height: 848 } },
    "3":  { class_type: "UnetLoaderGGUFAdvancedDisTorch2MultiGPU", inputs: { unet_name: "wan2.2-14B-Q6_K.gguf" } },
  }
  const r = parseComfyWorkflow(workflow)
  assert('positive',  r.positive,  "a cat on the moon")
  assert('negative',  r.negative,  "blurry, ugly")
  assert('steps',     r.steps,     6)
  assert('cfg',       r.cfg,       3.0)
  assert('sampler',   r.sampler,   "uni_pc")
  assert('scheduler', r.scheduler, "karras")
  assert('seed',      r.seed,      42)
  assert('width',     r.width,     480)
  assert('height',    r.height,    848)
  assert('model',     r.model,     "wan2.2-14B-Q6_K.gguf")
}

// ─── Test 2: No _meta.title - fallback via KSampler tracing ──────────────────
console.log('\n=== Test 2: Fallback trace (no _meta.title) ===')
{
  const workflow = {
    "1": { class_type: "CLIPTextEncode", inputs: { text: "beautiful forest" } },
    "2": { class_type: "CLIPTextEncode", inputs: { text: "worst quality" } },
    "3": { class_type: "KSampler", inputs: { positive: ["1",0], negative: ["2",0], steps: 20, cfg: 7.5, sampler_name: "dpmpp_2m", scheduler: "karras", seed: 99 } },
    "4": { class_type: "EmptyLatentImage", inputs: { width: 512, height: 512 } },
    "5": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "v1-5-pruned.safetensors" } },
  }
  const r = parseComfyWorkflow(workflow)
  assert('positive',  r.positive,  "beautiful forest")
  assert('negative',  r.negative,  "worst quality")
  assert('steps',     r.steps,     20)
  assert('cfg',       r.cfg,       7.5)
  assert('sampler',   r.sampler,   "dpmpp_2m")
  assert('scheduler', r.scheduler, "karras")
  assert('seed',      r.seed,      99)
  assert('width',     r.width,     512)
  assert('height',    r.height,    512)
  assert('model',     r.model,     "v1-5-pruned.safetensors")
}

// ─── Test 3: Empty / null input ───────────────────────────────────────────────
console.log('\n=== Test 3: Empty inputs ===')
assert('null input',  Object.keys(parseComfyWorkflow(null)).length,   0)
assert('empty obj',   Object.keys(parseComfyWorkflow({})).length,     0)

// ─── Test 4: Partial workflow (only positive, no advanced settings) ───────────
console.log('\n=== Test 4: Partial (only positive prompt) ===')
{
  const workflow = {
    "1": { class_type: "CLIPTextEncode", inputs: { text: "sunset" }, _meta: { title: "Positive Prompt" } },
  }
  const r = parseComfyWorkflow(workflow)
  assert('positive',  r.positive, "sunset")
  assert('no steps',  r.steps,    undefined)
  assert('no model',  r.model,    undefined)
}

// ─── Summary ──────────────────────────────────────────────────────────────────
console.log(`\n${'─'.repeat(40)}`)
console.log(`${passed + failed} checks: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
