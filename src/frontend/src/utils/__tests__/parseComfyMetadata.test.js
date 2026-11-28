/** Tests for the ComfyUI workflow-metadata parser (pure functions). */
import { describe, expect, it } from 'vitest'
import { parseComfyWorkflow } from '../parseComfyMetadata'

const node = (class_type, inputs = {}, meta = {}) => ({
  class_type,
  inputs,
  _meta: meta,
})

describe('input validation', () => {
  it('returns an empty object for null, non-objects and empty graphs', () => {
    expect(parseComfyWorkflow(null)).toEqual({})
    expect(parseComfyWorkflow('workflow')).toEqual({})
    expect(parseComfyWorkflow({})).toEqual({})
  })
})

describe('prompt extraction', () => {
  it('uses titled CLIPTextEncode nodes first (strategy A)', () => {
    const wf = {
      1: node('CLIPTextEncode', { text: 'a cat' }, { title: 'Positive Prompt' }),
      2: node('CLIPTextEncode', { text: 'blurry' }, { title: 'Negative Prompt' }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.positive).toBe('a cat')
    expect(r.negative).toBe('blurry')
  })

  it('traces untitled encodes from the primary KSampler (strategy B)', () => {
    const wf = {
      1: node('CLIPTextEncode', { text: 'a dog' }),
      2: node('CLIPTextEncode', { text: 'low res' }),
      3: node('KSampler', {
        seed: 7,
        steps: 24,
        cfg: 4.5,
        sampler_name: 'uni_pc',
        scheduler: 'karras',
        positive: ['1', 0],
        negative: ['2', 0],
        latent_image: ['10', 0],
      }),
      10: node('EmptySD3LatentImage', { width: 832, height: 480 }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.positive).toBe('a dog')
    expect(r.negative).toBe('low res')
    expect(r.steps).toBe(24)
    expect(r.cfg).toBe(4.5)
    expect(r.sampler).toBe('uni_pc')
    expect(r.scheduler).toBe('karras')
    expect(r.seed).toBe(7)
    expect(r.width).toBe(832)
    expect(r.height).toBe(480)
  })

  it('reads WanVideoTextEncodeMultiGPU prompts (strategy C)', () => {
    const wf = {
      1: node('WanVideoTextEncodeMultiGPU', {
        positive_prompt: 'sunset over water',
        negative_prompt: 'static',
      }),
    }
    expect(parseComfyWorkflow(wf).positive).toBe('sunset over water')
    expect(parseComfyWorkflow(wf).negative).toBe('static')
  })

  it('falls back to prompt-like keys on unknown node types (strategy D)', () => {
    const wf = {
      1: node('ExoticFutureNode', { text_positive: 'future positive' }),
    }
    expect(parseComfyWorkflow(wf).positive).toBe('future positive')
  })
})

describe('sampler settings fallbacks', () => {
  it('reads steps and scheduler from BasicScheduler and seed from RandomNoise (H3)', () => {
    const wf = {
      1: node('BasicScheduler', { steps: 8, scheduler: 'beta' }),
      2: node('RandomNoise', { noise_seed: 42 }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.steps).toBe(8)
    expect(r.scheduler).toBe('beta')
    expect(r.seed).toBe(42)
  })

  it('prefers noise_seed over seed on a KSamplerAdvanced', () => {
    const wf = {
      1: node('KSamplerAdvanced', { add_noise: 'enable', noise_seed: 11, seed: 99 }),
    }
    expect(parseComfyWorkflow(wf).seed).toBe(11)
  })

  it('reads fps from the VHS_VideoCombine node', () => {
    const wf = { 5: node('VHS_VideoCombine', { frame_rate: 16 }) }
    expect(parseComfyWorkflow(wf).fps).toBe(16)
  })
})

describe('dimensions', () => {
  it('takes width, height and length from the first known latent node', () => {
    const wf = {
      1: node('EmptyHunyuanLatentVideo', { width: 1280, height: 720, length: 97 }),
      2: node('EmptySD3LatentImage', { width: 640, height: 480 }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.width).toBe(1280)
    expect(r.height).toBe(720)
    expect(r.length).toBe(97)
  })
})

describe('model name', () => {
  it('reads ckpt_name from CheckpointLoaderSimple', () => {
    const wf = { 1: node('CheckpointLoaderSimple', { ckpt_name: 'wan_q6.gguf' }) }
    expect(parseComfyWorkflow(wf).model).toBe('wan_q6.gguf')
  })

  it('reads unet_name from UnetLoaderGGUF', () => {
    const wf = { 1: node('UnetLoaderGGUF', { unet_name: 'h3_fp8.safetensors' }) }
    expect(parseComfyWorkflow(wf).model).toBe('h3_fp8.safetensors')
  })

  it('drops the model when the loader value is a reference, not a filename', () => {
    const wf = {
      1: node('CheckpointLoaderSimple', { ckpt_name: ['9', 0] }),
      9: node('AnythingElse', {}),
    }
    expect(parseComfyWorkflow(wf).model).toBeUndefined()
  })
})

describe('lora extraction', () => {
  it('pairs two Power Lora Loaders as high and low noise', () => {
    const wf = {
      100: node('Power Lora Loader (rgthree)', {
        lora_1: { on: true, lora: 'detail_h.safetensors', strength: 0.8 },
        lora_2: { on: false, lora: 'off_h.safetensors', strength: 0.9 },
      }),
      110: node('Power Lora Loader (rgthree)', {
        lora_1: { on: true, lora: 'detail_l.safetensors', strength: 0.5 },
      }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.loras).toEqual([
      { high: 'detail_h.safetensors', low: 'detail_l.safetensors', strength: 0.8 },
    ])
  })

  it('uses one Power Lora Loader for both slots', () => {
    const wf = {
      100: node('Power Lora Loader (rgthree)', {
        lora_1: { on: true, lora: 'solo.safetensors', strength: 1.1 },
      }),
    }
    expect(parseComfyWorkflow(wf).loras).toEqual([
      { high: 'solo.safetensors', low: 'solo.safetensors', strength: 1.1 },
    ])
  })

  it('pairs LoraLoaderModelOnly chains by the 170/180 id ranges', () => {
    const wf = {
      171: node('LoraLoaderModelOnly', { lora_name: 'hi_a.safetensors', strength_model: 0.7 }),
      181: node('LoraLoaderModelOnly', { lora_name: 'lo_a.safetensors', strength_model: 0.6 }),
    }
    expect(parseComfyWorkflow(wf).loras).toEqual([
      { high: 'hi_a.safetensors', low: 'lo_a.safetensors', strength: 0.7 },
    ])
  })

  it('treats out-of-range single loaders as high-only, skipping bundled turbo loras', () => {
    const wf = {
      170: node('LoraLoaderModelOnly', { lora_name: 'hi_a.safetensors', strength_model: 0.7 }),
      190: node('LoraLoaderModelOnly', { lora_name: 'user_style.safetensors', strength_model: 0.9 }),
      // strength present on purpose: proves the filter is name-based,
      // not strength-based (PR-Piet 2026-11-28 confound)
      191: node('LoraLoaderModelOnly', { lora_name: 'minimax_h3_fl2v_turbo_v1.safetensors', strength_model: 0.8 }),
    }
    const r = parseComfyWorkflow(wf)
    // 170 pairs with an empty 180-slot (low falls back to its own high name);
    // 190 is out of both ranges -> high-only extra entry; turbo lora filtered.
    expect(r.loras).toEqual([
      { high: 'hi_a.safetensors', low: 'hi_a.safetensors', strength: 0.7 },
      { high: 'user_style.safetensors', low: '', strength: 0.9 },
    ])
  })

  it('returns no loras key for a workflow without loaders', () => {
    expect(parseComfyWorkflow({ 1: node('KSampler', {}) }).loras).toBeUndefined()
  })
})

describe('strategy precedence', () => {
  it('a titled encode beats the KSampler trace (A over B)', () => {
    const wf = {
      1: node('CLIPTextEncode', { text: 'titled pos' }, { title: 'Positive Prompt' }),
      2: node('CLIPTextEncode', { text: 'traced pos' }),
      3: node('KSampler', { positive: ['2', 0], steps: 20 }),
    }
    expect(parseComfyWorkflow(wf).positive).toBe('titled pos')
  })

  it('the KSampler trace beats the WanVideo fallback for positive (B over C)', () => {
    const wf = {
      1: node('CLIPTextEncode', { text: 'traced pos' }),
      2: node('KSampler', { positive: ['1', 0] }),
      3: node('WanVideoTextEncodeMultiGPU', { positive_prompt: 'wan pos', negative_prompt: 'wan neg' }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.positive).toBe('traced pos')
    expect(r.negative).toBe('wan neg')   // B fills only its gaps; C supplies the rest
  })

  it('the KSampler trace follows the graph, not id order (B is tracing)', () => {
    const wf = {
      1: node('CLIPTextEncode', { text: 'decoy first' }),
      2: node('CLIPTextEncode', { text: 'real positive' }),
      3: node('KSampler', { positive: ['2', 0] }),
    }
    expect(parseComfyWorkflow(wf).positive).toBe('real positive')
  })

  it('a real KSampler beats the BasicScheduler fallback for steps and scheduler', () => {
    const wf = {
      1: node('KSampler', { steps: 24, cfg: 5, sampler_name: 'uni_pc', scheduler: 'karras' }),
      2: node('BasicScheduler', { steps: 8, scheduler: 'beta' }),
    }
    const r = parseComfyWorkflow(wf)
    expect(r.steps).toBe(24)
    expect(r.scheduler).toBe('karras')
  })

  it('the WanVideo node beats the generic key fallback (C over D)', () => {
    const wf = {
      1: node('WanVideoTextEncodeMultiGPU', { positive_prompt: 'wan wins' }),
      2: node('ExoticNode', { text_positive: 'generic loses' }),
    }
    expect(parseComfyWorkflow(wf).positive).toBe('wan wins')
  })
})
