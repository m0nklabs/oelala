import sys, os
sys.path.append(os.path.abspath('src'))
from backend.comfyui_client import comfyui
w = comfyui.build_cloud_ltx23_i2v_workflow(image_name="input.png", prompt="test", num_frames=41)
import json
print(json.dumps(w, indent=2))
