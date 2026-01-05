/**
 * Default prompts for new users.
 * These are shown when no localStorage prompt exists.
 *
 * Format: Each prompt should describe a visually interesting,
 * family-friendly animation that works well with Image-to-Video.
 */

export const SFW_PROMPTS = [
  // Nature & Landscapes
  "Gentle waves lapping on a tropical beach at sunset, palm trees swaying softly in the breeze, golden hour light reflecting on the water",
  "A serene mountain lake with crystal clear water, surrounded by snow-capped peaks, subtle ripples from a gentle breeze",
  "Cherry blossoms falling gracefully in a Japanese garden, petals dancing in the wind, soft spring sunlight",
  "Northern lights dancing across an Arctic sky, vibrant greens and purples, stars twinkling in the background",
  "A misty forest at dawn, sunbeams breaking through the canopy, dew drops glistening on leaves",

  // Urban & Architecture
  "A cozy coffee shop window on a rainy day, raindrops sliding down the glass, warm light from inside",
  "Tokyo cityscape at night, neon signs flickering, light trails from passing cars",
  "A quaint European village street, autumn leaves blowing, warm café lights in windows",
  "Modern architecture with flowing water features, reflections dancing on glass surfaces",

  // Abstract & Artistic
  "Colorful ink drops spreading in water, hypnotic swirling patterns, smooth organic motion",
  "Floating soap bubbles catching rainbow light, drifting slowly through a sunlit room",
  "Abstract fluid art in motion, vibrant colors mixing and flowing, mesmerizing patterns",
  "Geometric shapes morphing and transforming, satisfying transitions, clean minimalist style",

  // People & Lifestyle (SFW)
  "A skilled barista pouring latte art, steam rising from the cup, focused concentration",
  "A dancer gracefully spinning, flowing fabric catching the light, elegant movement",
  "An artist painting on a large canvas, bold brushstrokes, creative energy",
  "A chef plating an exquisite dish, precise movements, steam rising from the food",

  // Animals & Nature
  "A majestic eagle soaring through mountain clouds, wings spread wide, powerful and free",
  "Colorful koi fish swimming in a crystal clear pond, graceful movements, dappled sunlight",
  "A fluffy cat stretching lazily by a sunny window, content and peaceful",
  "Butterflies dancing around a blooming flower garden, vibrant colors, gentle flight",

  // Fantasy & Sci-Fi (SFW)
  "A magical portal opening with swirling energy, mystical light emanating, otherworldly glow",
  "A futuristic cityscape with flying vehicles, holographic advertisements, sleek architecture",
  "An enchanted forest with glowing mushrooms, fireflies dancing, magical atmosphere",
  "A steampunk clockwork mechanism turning, brass gears rotating, intricate details",
]

export const NSFW_PROMPTS = [
  // These would be adult prompts - only shown when logged in with NSFW enabled
  // Keeping empty for now - to be populated based on content policy
]

/**
 * Get a random default prompt based on NSFW setting
 */
export function getRandomPrompt(nsfwEnabled = false) {
  const pool = nsfwEnabled ? [...SFW_PROMPTS, ...NSFW_PROMPTS] : SFW_PROMPTS
  const randomIndex = Math.floor(Math.random() * pool.length)
  return pool[randomIndex]
}

/**
 * Get a prompt - returns saved prompt if exists, otherwise random default
 */
export function getDefaultPrompt(nsfwEnabled = false) {
  try {
    const saved = localStorage.getItem('oelala_last_prompt')
    if (saved && saved.trim()) {
      return saved
    }
  } catch {
    // localStorage not available
  }
  return getRandomPrompt(nsfwEnabled)
}
