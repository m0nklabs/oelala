export const TOOL_IDS = {
  TEXT_TO_IMAGE: 'text-to-image',
  TEXT_TO_VIDEO: 'text-to-video',
  IMAGE_TO_VIDEO: 'image-to-video',
  TEXT_TO_IMAGE_TO_VIDEO: 'text-to-image-to-video',
  PIPELINE: 'pipeline',
  VIDEO_TO_VIDEO: 'video-to-video',
  IMAGE_TO_IMAGE: 'image-to-image',
  REFRAME: 'reframe',
  FACE_SWAP: 'face-swap',
  UPSCALER: 'upscaler',
  LORA_TRAINING: 'lora-training',
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
      { id: TOOL_IDS.TEXT_TO_VIDEO, label: 'Text to Video', status: 'soon' },
      { id: TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO, label: 'Text to Image to Video', status: 'new' },
      { id: TOOL_IDS.VIDEO_TO_VIDEO, label: 'Video to Video', status: 'missing-backend' },
    ],
  },
  {
    id: 'image-tools',
    title: 'Image Tools',
    items: [
      { id: TOOL_IDS.TEXT_TO_IMAGE, label: 'Text to Image', status: 'missing-backend' },
      { id: TOOL_IDS.IMAGE_TO_IMAGE, label: 'Image to Image', status: 'missing-backend' },
      { id: TOOL_IDS.REFRAME, label: 'Reframe', status: 'missing-backend' },
      { id: TOOL_IDS.FACE_SWAP, label: 'Face Swap', status: 'missing-backend' },
      { id: TOOL_IDS.UPSCALER, label: 'Upscaler', status: 'missing-backend' },
    ],
  },
  {
    id: 'advanced',
    title: 'Advanced',
    items: [
      { id: TOOL_IDS.PIPELINE, label: 'Pipeline', status: 'new' },
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
      { id: TOOL_IDS.MY_MEDIA_PROMPTS, label: 'Prompts', status: 'new' },
    ],
  },
]
