#!/usr/bin/env python3
"""Quick H.264 encoding test for placeholder frames.

Generates simple gradient frames and uses Wan2VideoGenerator.save_video
to ensure ffmpeg/libx264 path works. Does NOT load heavy models.
Run: python test_h264_encode.py
"""

import os
import subprocess
import numpy as np
from PIL import Image

def build_frames(num_frames: int = 8, width: int = 832, height: int = 480):
    frames = []
    for i in range(num_frames):
        base = np.zeros((height, width, 3), dtype=np.uint8)
        # Simple moving gradient
        base[:, :, 0] = (np.linspace(0, 255, width)[None, :] + i * 10) % 256
        base[:, :, 1] = (np.linspace(255, 0, width)[None, :] + i * 5) % 256
        base[:, :, 2] = ((i * 30) % 256)
        img = Image.fromarray(base, 'RGB')
        frames.append(img)
    return frames

def main():
    out_dir = "/home/flip/oelala/generated"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "h264_test_placeholder.mp4")
    frames = build_frames()
    print(f"🔧 Creating test video: {output_path}")
    # Encode via ffmpeg directly (libx264 yuv420p)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="h264_frames_")
    for i, frame in enumerate(frames):
        frame.save(os.path.join(tmp, f"frame_{i:05d}.png"))
    cmd = [
        'ffmpeg','-y','-framerate','8','-i',os.path.join(tmp,'frame_%05d.png'),
        '-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',
        output_path
    ]
    print('🚀 Encoding:', ' '.join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("✅ H.264 test video created:", output_path)
    print(f"Run: ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height -of json {output_path}")

if __name__ == "__main__":
    main()
