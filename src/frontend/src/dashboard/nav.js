export const TOOL_IDS = {
  // Video Tools
  TEXT_TO_IMAGE: 'text-to-image',
  TEXT_TO_VIDEO: 'text-to-video',
  IMAGE_TO_VIDEO: 'image-to-video',
  TEXT_TO_IMAGE_TO_VIDEO: 'text-to-image-to-video',
  VIDEO_TO_VIDEO: 'video-to-video',
  VIDEO_TO_TEXT: 'video-to-text',
  // Image Tools
  IMAGE_TO_IMAGE: 'image-to-image',
  REFRAME: 'reframe',
  FACE_SWAP: 'face-swap',
  UPSCALER: 'upscaler',
  // Prompt Tools
  IMAGE_TO_TEXT: 'image-to-text',
  PROMPT_GENERATOR: 'prompt-generator',
  // Audio Tools
  AUDIO_GENERATION: 'audio-generation',
  VOICE_CLONING: 'voice-cloning',
  LIP_SYNC: 'lip-sync',
  // Advanced
  PIPELINE: 'pipeline',
  LORA_TRAINING: 'lora-training',
  // My Media
  MY_MEDIA_ALL: 'my-media-all',
  MY_MEDIA_VIDEOS: 'my-media-videos',
  MY_MEDIA_IMAGES: 'my-media-images',
  MY_MEDIA_PROMPTS: 'my-media-prompts',
}

export const NAV_GROUPS = [
  {
    id: 'video-tools',
    title: 'Video Tools',
    items: [
      { id: TOOL_IDS.IMAGE_TO_VIDEO, label: 'Image to Video', status: 'ready' },
      { id: TOOL_IDS.TEXT_TO_VIDEO, label: 'Text to Video', status: 'ready' },
      { id: TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO, label: 'Text to Image to Video', status: 'ready' },
      { id: TOOL_IDS.VIDEO_TO_VIDEO, label: 'Video to Video', status: 'ready' },
      { id: TOOL_IDS.VIDEO_TO_TEXT, label: 'Video to Text', status: 'new' },
    ],
  },
  {
    id: 'image-tools',
    title: 'Image Tools',
    items: [
      { id: TOOL_IDS.TEXT_TO_IMAGE, label: 'Text to Image', status: 'ready' },
      { id: TOOL_IDS.IMAGE_TO_IMAGE, label: 'Image to Image', status: 'ready' },
      { id: TOOL_IDS.UPSCALER, label: 'Upscaler', status: 'ready' },
      { id: TOOL_IDS.REFRAME, label: 'Reframe', status: 'new' },
      { id: TOOL_IDS.FACE_SWAP, label: 'Face Swap', status: 'new' },
    ],
  },
  {
    id: 'prompt-tools',
    title: 'Prompt Tools',
    items: [
      { id: TOOL_IDS.IMAGE_TO_TEXT, label: 'Image to Text', status: 'new' },
      { id: TOOL_IDS.PROMPT_GENERATOR, label: 'Prompt Generator', status: 'new' },
    ],
  },
  {
    id: 'audio-tools',
    title: 'Audio Tools',
    items: [
      { id: TOOL_IDS.AUDIO_GENERATION, label: 'Audio Generation', status: 'new' },
      { id: TOOL_IDS.VOICE_CLONING, label: 'Voice Cloning', status: 'new' },
      { id: TOOL_IDS.LIP_SYNC, label: 'Lip Sync', status: 'new' },
    ],
  },
  {
    id: 'advanced',
    title: 'Advanced',
    items: [
      { id: TOOL_IDS.PIPELINE, label: 'Pipeline', status: 'ready' },
      { id: TOOL_IDS.LORA_TRAINING, label: 'LoRA Training', status: 'ready' },
    ],
  },
  {
    id: 'my-media',
    title: 'My Media',
    items: [
      { id: TOOL_IDS.MY_MEDIA_ALL, label: 'All', status: 'ready' },
      { id: TOOL_IDS.MY_MEDIA_VIDEOS, label: 'Videos', status: 'ready' },
      { id: TOOL_IDS.MY_MEDIA_IMAGES, label: 'Images', status: 'ready' },
      { id: TOOL_IDS.MY_MEDIA_PROMPTS, label: 'Prompts', status: 'ready' },
    ],
  },
]
