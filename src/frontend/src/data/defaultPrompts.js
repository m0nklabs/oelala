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
  // Sensual / Artistic
  "A confident woman posing in elegant lingerie, soft studio lighting, silk sheets, artistic boudoir photography style",
  "A couple sharing an intimate slow dance, close embrace, candlelight atmosphere, romantic mood",
  "A woman undressing slowly by a rain-streaked window, city lights behind her, cinematic shadows",
  "Soft romantic scene, two lovers intertwined on white linen, golden hour light, tender intimacy",
  "A sensual model in sheer fabric, wind blowing softly, dramatic backlit silhouette",

  // Fashion / Adult editorial
  "High-fashion editorial shoot, model in revealing designer outfit, bold dramatic lighting, fierce energy",
  "A glamorous woman in a sheer evening gown, champagne glass in hand, rooftop penthouse setting",
  "Artistic nude photography, female figure, black and white, strong side lighting, museum quality",
  "A seductive burlesque performer, feather boa, stage lights, confident playful expression",

  // Fantasy / Sci-Fi (adult)
  "A powerful succubus emerging from smoke, revealing dark fantasy armor, otherworldly glow, fierce eyes",
  "An ethereal fae queen in minimal gossamer robes, enchanted forest, magic light surrounding her",
  "A cyberpunk femme fatale in tight tech wear, neon-lit rain-soaked alley, fierce and dangerous",
  "A warrior goddess, minimal ceremonial armor, mythological setting, wind in her hair, commanding presence",

  // Mood / atmosphere
  "Two people in post-coital serenity, tangled sheets, morning light, peaceful and intimate",
  "A woman in a steamy outdoor hot tub, night sky above, relaxed expression, steam rising around her",
  "A bold confident woman in an open-back dress posing before a hotel mirror, luxury setting",
  "Vintage Playboy-style centerfold aesthetic, soft-focus 70s lighting, confident relaxed pose, tasteful",
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
