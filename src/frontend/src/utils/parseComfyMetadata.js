/**
 * Parse a raw ComfyUI API-format workflow JSON and extract human-useful settings.
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
    if (!text) continue
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
  // 2. Sampler settings — from the "primary" (first-pass) sampler
  // ─────────────────────────────────────────────────────────────────
  const sampler = _findPrimarySampler(nodes)
  if (sampler) {
    const inp = sampler.inputs || {}
    if (inp.steps !== undefined)       result.steps     = inp.steps
    if (inp.cfg   !== undefined)       result.cfg       = inp.cfg
    if (inp.sampler_name)              result.sampler   = inp.sampler_name
    if (inp.scheduler)                 result.scheduler = inp.scheduler
    if (inp.noise_seed !== undefined)  result.seed      = inp.noise_seed
    else if (inp.seed !== undefined)   result.seed      = inp.seed
  }

  // ─────────────────────────────────────────────────────────────────
  // 3. Dimensions — from latent / video nodes
  // ─────────────────────────────────────────────────────────────────
  const LATENT_TYPES = new Set([
    'EmptyLatentImage', 'EmptySD3LatentImage', 'EmptyHunyuanLatentVideo',
    'EmptyMochiLatentVideo', 'EmptyLTXVLatentVideo', 'WanImageToVideo',
    'WanFirstAndLastFrameToVideo',
  ])
  for (const [, node] of Object.entries(nodes)) {
    if (!LATENT_TYPES.has(node.class_type)) continue
    const inp = node.inputs || {}
    if (inp.width)  result.width  = inp.width
    if (inp.height) result.height = inp.height
    break
  }

  // ─────────────────────────────────────────────────────────────────
  // 4. Model name
  // ─────────────────────────────────────────────────────────────────
  const LOADER_TYPES = new Set([
    'CheckpointLoaderSimple', 'CheckpointLoader',
    'UnetLoaderGGUF', 'UnetLoaderGGUFAdvanced',
    'UnetLoaderGGUFAdvancedDisTorch2MultiGPU',
  ])
  for (const [, node] of Object.entries(nodes)) {
    if (!LOADER_TYPES.has(node.class_type)) continue
    result.model = node.inputs?.ckpt_name || node.inputs?.unet_name
    break
  }

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

/**
 * Follow a node-link chain until we reach a CLIPTextEncode and return its text.
 * Handles intermediate nodes (e.g. WanImageToVideo) by following the same
 * "positive" / "negative" input key further up the graph.
 */
function _traceToText(nodes, startNodeId, inputKey, depth = 0) {
  if (depth > 6 || !startNodeId) return null
  const node = nodes[String(startNodeId)]
  if (!node) return null

  if (node.class_type === 'CLIPTextEncode') return node.inputs?.text ?? null

  // Follow the same key deeper
  const nextRef = node.inputs?.[inputKey]
  if (Array.isArray(nextRef)) return _traceToText(nodes, nextRef[0], inputKey, depth + 1)

  return null
}
