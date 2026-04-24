### Fixed
- Fixed bug where video files in the "My Creations" gallery failed to show generative tools in the "Use in tool" dropdown due to a stale deployed frontend build.
- Fixed `parseComfyMetadata.js` so it now extracts `audio` and `audio_prompt` from ComfyUI video metadata (specifically supporting LTXV and VividAudioPrompt node metadata), allowing users to import audio prompts from previous generative videos.
