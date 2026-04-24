/**
 * Parse a raw ComfyUI API-format workflow JSON and extract human-useful settings.
 *
 * Supports:
 *  - Standard ComfyUI workflows (CLIPTextEncode + KSampler)
 *  - DisTorch2 multi-GPU workflows
 *  - Kijai WanVideoWrapper multi-GPU workflows (WanVideoTextEncodeMultiGPU etc.)
 *  - Generic fallback for unknown workflow types
 *
 * Returns an object with any of:
 *   positive  : string  - positive prompt text
 *   negative  : string  - negative prompt text
 *   steps     : number  - sampling steps
 *   cfg       : number  - CFG / guidance scale
 *   sampler   : string  - sampler name (e.g. "uni_pc", "dpmpp_2m")
 *   scheduler : string  - scheduler (e.g. "karras", "normal")
 *   seed      : number  - noise seed
 *   width     : number  - output width
 *   height    : number  - output height
 *   model     : string  - checkpoint / unet filename
 */
export function parseComfyWorkflow(workflow) {
  if (!workflow || typeof workflow !== 'object') return {}

  const result = {}
  const nodes = workflow // workflow IS the node map: { "1": { class_type, inputs, _meta }, ... }

  // ─────────────────────────────────────────────────────────────────
  // 1. Find positive / negative prompts
  //    Strategy A: CLIPTextEncode whose _meta.title contains pos/neg
  // ─────────────────────────────────────────────────────────────────
  for (const [, node] of Object.entries(nodes)) {
    if (node.class_type !== 'CLIPTextEncode') continue
    const text = node.inputs?.text
    if (!text || typeof text !== 'string') continue
    const title = (node._meta?.title || '').toLowerCase()
    if (!result.positive && (title.includes('positive') || title === 'pos')) {
      result.positive = text
    } else if (!result.negative && (title.includes('negative') || title === 'neg')) {
      result.negative = text
    }
  }

  // ─────────────────────────────────────────────────────────────────
  //    Strategy B: trace from the primary KSampler node
  // ─────────────────────────────────────────────────────────────────
  if (!result.positive || !result.negative) {
    const sampler = _findPrimarySampler(nodes)
    if (sampler) {
      const inputs = sampler.inputs || {}
      if (!result.positive && Array.isArray(inputs.positive)) {
        result.positive = _traceToText(nodes, inputs.positive[0], 'positive')
      }
      if (!result.negative && Array.isArray(inputs.negative)) {
        result.negative = _traceToText(nodes, inputs.negative[0], 'negative')
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────
  //    Strategy C: Kijai WanVideoWrapper & LTXV nodes
  //    (WanVideoTextEncodeMultiGPU has positive_prompt / negative_prompt, 
  //     LTXVCPUGemmaEncode has text)
  // ─────────────────────────────────────────────────────────────────
  if (!result.positive || !result.negative || !result.audio) {
    for (const [, node] of Object.entries(nodes)) {
      // WanVideo
      if (node.class_type === 'WanVideoTextEncodeMultiGPU') {
        const inp = node.inputs || {}
        if (!result.positive && typeof inp.positive_prompt === 'string') {
          result.positive = inp.positive_prompt
        }
        if (!result.negative && typeof inp.negative_prompt === 'string') {
          result.negative = inp.negative_prompt
        }
      }
      
      // LTXV
      if (node.class_type === 'LTXVCPUGemmaEncode') {
        const inp = node.inputs || {}
        if (typeof inp.text === 'string') {
          if (!result.positive) result.positive = inp.text
          if (!result.audio) result.audio = inp.text
        }
      }
      if (node.class_type === 'LTXVCPUGemmaNegativeEncode') {
        const inp = node.inputs || {}
        if (!result.negative && typeof inp.text === 'string') {
          result.negative = inp.text
        }
      }
      
      // Vivid Audio Prompt
      if (node.class_type === 'VividAudioPrompt') {
        const inp = node.inputs || {}
        if (!result.audio && typeof inp.prompt === 'string') {
          result.audio = inp.prompt
        }
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────
  //    Strategy D: Generic fallback — scan all nodes for prompt-like
  //    text input keys (works for unknown/future node types)
  // ─────────────────────────────────────────────────────────────────
  if (!result.positive || !result.negative || !result.audio) {
    const POS_KEYS = ['positive_prompt', 'text_positive', 'prompt', 'text', 'text_g']
    const NEG_KEYS = ['negative_prompt', 'text_negative', 'text_l']
    const AUDIO_KEYS = ['audio_prompt', 'audio']
    for (const [, node] of Object.entries(nodes)) {
      const inp = node.inputs || {}
      for (const [key, value] of Object.entries(inp)) {
        if (typeof value !== 'string' || value.length < 5) continue
        const k = key.toLowerCase()
        if (!result.positive && POS_KEYS.includes(k)) result.positive = value
        if (!result.negative && NEG_KEYS.includes(k)) result.negative = value
        if (!result.audio && AUDIO_KEYS.includes(k)) result.audio = value
      }
      if (result.positive && result.negative && result.audio) break
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // 2. Sampler settings — from the "primary" (first-pass) sampler
  //    Checks standard KSampler first, then WanVideoSamplerMultiGPU
  // ─────────────────────────────────────────────────────────────────
  const sampler = _findPrimarySampler(nodes) || _findWanVideoSampler(nodes)
  if (sampler) {
    const inp = sampler.inputs || {}
    if (typeof inp.steps === 'number')        result.steps     = inp.steps
    if (typeof inp.cfg === 'number')          result.cfg       = inp.cfg
    if (typeof inp.sampler_name === 'string') result.sampler   = inp.sampler_name
    if (typeof inp.scheduler === 'string')    result.scheduler = inp.scheduler
    if (typeof inp.noise_seed === 'number')   result.seed      = inp.noise_seed
    else if (typeof inp.seed === 'number')    result.seed      = inp.seed
  }

  // ─────────────────────────────────────────────────────────────────
  // 3. Dimensions — from latent / video nodes
  // ─────────────────────────────────────────────────────────────────
  const LATENT_TYPES = new Set([
    'EmptyLatentImage', 'EmptySD3LatentImage', 'EmptyHunyuanLatentVideo',
    'EmptyMochiLatentVideo', 'EmptyLTXVLatentVideo', 'WanImageToVideo',
    'WanFirstAndLastFrameToVideo', 'WanVideoImageToVideoEncodeMultiGPU',
  ])
  for (const [, node] of Object.entries(nodes)) {
    if (!LATENT_TYPES.has(node.class_type)) continue
    const inp = node.inputs || {}
    if (typeof inp.width === 'number')  result.width  = inp.width
    if (typeof inp.height === 'number') result.height = inp.height
    break
  }

  // ─────────────────────────────────────────────────────────────────
  // 4. Model name
  // ─────────────────────────────────────────────────────────────────
  const LOADER_TYPES = new Set([
    'CheckpointLoaderSimple', 'CheckpointLoader',
    'UnetLoaderGGUF', 'UnetLoaderGGUFAdvanced',
    'UnetLoaderGGUFAdvancedDisTorch2MultiGPU',
    'WanVideoModelLoaderMultiGPU',
  ])
  for (const [, node] of Object.entries(nodes)) {
    if (!LOADER_TYPES.has(node.class_type)) continue
    result.model = node.inputs?.ckpt_name || node.inputs?.unet_name || node.inputs?.model
    if (typeof result.model === 'string') break
    delete result.model // was an array ref, not a filename
  }

  // ─────────────────────────────────────────────────────────────────
  // 5. LoRA configs — extract from Power Lora Loader (rgthree) or
  //    LoraLoaderModelOnly chains
  // ─────────────────────────────────────────────────────────────────
  const loras = _extractLoras(nodes)
  if (loras.length > 0) result.loras = loras

  return result
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const SAMPLER_TYPES = new Set(['KSampler', 'KSamplerAdvanced', 'SamplerCustom'])

/** Return the "primary" sampler: the KSamplerAdvanced with add_noise=enable,
 *  or the first KSampler if no advanced nodes exist. */
function _findPrimarySampler(nodes) {
  let fallback = null
  for (const [, node] of Object.entries(nodes)) {
    if (!SAMPLER_TYPES.has(node.class_type)) continue
    if (node.class_type === 'KSamplerAdvanced') {
      if (node.inputs?.add_noise === 'enable') return node
    } else {
      fallback = fallback ?? node
    }
  }
  return fallback
}

/** Return the primary WanVideoSamplerMultiGPU: the one with denoise_strength=1
 *  or the first one without a samples input (= first-pass, not refinement). */
function _findWanVideoSampler(nodes) {
  let fallback = null
  for (const [, node] of Object.entries(nodes)) {
    if (node.class_type !== 'WanVideoSamplerMultiGPU') continue
    const inp = node.inputs || {}
    // Primary sampler: denoise 1.0, no samples input (not a refinement pass)
    if (inp.denoise_strength === 1 || inp.denoise_strength === 1.0) return node
    if (!inp.samples) return node
    fallback = fallback ?? node
  }
  return fallback
}

/**
 * Follow a node-link chain until we reach a CLIPTextEncode and return its text.
 * Handles intermediate nodes (e.g. WanImageToVideo) by following the same
 * "positive" / "negative" input key further up the graph.
 *
 * When CLIPTextEncode.text is a reference (array) instead of a string, we
 * resolve it via _resolveTextRef, which can trace through StringConcatenate,
 * Text Find and Replace, and similar text-manipulation nodes.
 */
function _traceToText(nodes, startNodeId, inputKey, depth = 0) {
  if (depth > 6 || !startNodeId) return null
  const node = nodes[String(startNodeId)]
  if (!node) return null

  if (node.class_type === 'CLIPTextEncode') {
    const text = node.inputs?.text
    if (typeof text === 'string') return text
    // text is a reference (e.g. ['451', 0]) — resolve it
    if (Array.isArray(text)) return _resolveTextRef(nodes, text[0])
    return null
  }

  // Follow the same key deeper
  const nextRef = node.inputs?.[inputKey]
  if (Array.isArray(nextRef)) return _traceToText(nodes, nextRef[0], inputKey, depth + 1)

  return null
}

/**
 * Resolve a text reference chain through text-manipulation nodes.
 * Handles: StringConcatenate, Text Find and Replace, and other nodes
 * that produce text output from text inputs.
 *
 * For StringConcatenate: returns string_b preferentially (user prompt),
 * falling back to string_a (often auto-caption), with delimiter between
 * if both are resolvable strings.
 */
function _resolveTextRef(nodes, nodeId, depth = 0) {
  if (depth > 12 || !nodeId) return null
  const node = nodes[String(nodeId)]
  if (!node) return null

  const ct = node.class_type || ''
  const inp = node.inputs || {}

  // StringConcatenate: has string_a, string_b, delimiter
  if (ct === 'StringConcatenate') {
    const a = _resolveInputText(nodes, inp.string_a, depth)
    const b = _resolveInputText(nodes, inp.string_b, depth)
    const delim = typeof inp.delimiter === 'string' ? inp.delimiter : ' '
    if (a && b) return a + delim + b
    return b || a || null
  }

  // Text Find and Replace: passes text through with replacements
  if (ct === 'Text Find and Replace') {
    return _resolveInputText(nodes, inp.text, depth)
  }

  // CLIPTextEncode hit again (shouldn't happen, but handle it)
  if (ct === 'CLIPTextEncode') {
    return _resolveInputText(nodes, inp.text, depth)
  }

  // Generic: try common text output keys
  for (const key of ['text', 'string', 'output', 'result', 'text_output']) {
    const val = inp[key]
    if (typeof val === 'string' && val.length > 0) return val
    if (Array.isArray(val)) {
      const resolved = _resolveTextRef(nodes, val[0], depth + 1)
      if (resolved) return resolved
    }
  }

  return null
}

/** Resolve a single input value: if string return it, if array ref follow it. */
function _resolveInputText(nodes, value, depth) {
  if (typeof value === 'string' && value.length > 0) return value
  if (Array.isArray(value)) return _resolveTextRef(nodes, value[0], depth + 1)
  return null
}

/**
 * Extract LoRA configurations from the workflow.
 *
 * Supports:
 *  - Power Lora Loader (rgthree): paired high/low noise nodes with lora_1..lora_6 slots
 *  - LoraLoaderModelOnly chains: sequential nodes starting at id 170 (high) / 180 (low)
 *
 * Returns: Array of { high: string, low: string, strength: number }
 */
function _extractLoras(nodes) {
  const loras = []

  // Strategy A: Power Lora Loader (rgthree) — used by BlockSwap/DisTorch2 Q8 workflows
  const powerLoaders = []
  for (const [id, node] of Object.entries(nodes)) {
    if (node.class_type === 'Power Lora Loader (rgthree)') {
      powerLoaders.push({ id, inputs: node.inputs || {} })
    }
  }

  if (powerLoaders.length >= 2) {
    // Pair them: first = high noise, second = low noise (by node id order)
    powerLoaders.sort((a, b) => Number(a.id) - Number(b.id))
    const highInputs = powerLoaders[0].inputs
    const lowInputs = powerLoaders[1].inputs

    for (let i = 1; i <= 6; i++) {
      const highSlot = highInputs[`lora_${i}`]
      const lowSlot = lowInputs[`lora_${i}`]
      if (!highSlot?.on && !lowSlot?.on) continue
      const highName = (highSlot?.on && highSlot?.lora && highSlot.lora !== 'None') ? highSlot.lora : ''
      const lowName = (lowSlot?.on && lowSlot?.lora && lowSlot.lora !== 'None') ? lowSlot.lora : ''
      if (!highName && !lowName) continue
      loras.push({
        high: highName,
        low: lowName,
        strength: highSlot?.strength ?? lowSlot?.strength ?? 1.0,
      })
    }
    return loras
  }

  if (powerLoaders.length === 1) {
    // Single Power Lora Loader — use same for both high and low
    const inputs = powerLoaders[0].inputs
    for (let i = 1; i <= 6; i++) {
      const slot = inputs[`lora_${i}`]
      if (!slot?.on || !slot?.lora || slot.lora === 'None') continue
      loras.push({ high: slot.lora, low: slot.lora, strength: slot.strength ?? 1.0 })
    }
    return loras
  }

  // Strategy B: LoraLoaderModelOnly chains (used by T2V Q6 / I2V DisTorch2 workflows)
  const loraModelOnlyNodes = []
  for (const [id, node] of Object.entries(nodes)) {
    if (node.class_type === 'LoraLoaderModelOnly') {
      loraModelOnlyNodes.push({ id: Number(id), inputs: node.inputs || {} })
    }
  }

  if (loraModelOnlyNodes.length > 0) {
    // Group by range: 170-179 = high noise, 180-189 = low noise
    const highNodes = loraModelOnlyNodes.filter(n => n.id >= 170 && n.id < 180).sort((a, b) => a.id - b.id)
    const lowNodes = loraModelOnlyNodes.filter(n => n.id >= 180 && n.id < 190).sort((a, b) => a.id - b.id)
    const maxLen = Math.max(highNodes.length, lowNodes.length)

    for (let i = 0; i < maxLen; i++) {
      const highName = highNodes[i]?.inputs?.lora_name || ''
      const lowName = lowNodes[i]?.inputs?.lora_name || ''
      const strength = highNodes[i]?.inputs?.strength_model ?? lowNodes[i]?.inputs?.strength_model ?? 1.0
      if (highName || lowName) {
        loras.push({ high: highName, low: lowName || highName, strength })
      }
    }
  }

  return loras
}
