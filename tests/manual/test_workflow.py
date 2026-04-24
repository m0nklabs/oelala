import sys
from backend.comfyui_client import comfyui
from backend.types import GenerationRequest
w = comfyui.build_cloud_ltx23_i2v_workflow(image_name="input.png", prompt="test", num_frames=41)
import json
print(json.dumps(w, indent=2))
