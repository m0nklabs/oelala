import os
import re

files = [
    "/home/flip/oelala/src/backend/generation/adapters/cloud/cloud_i2i.py",
    "/home/flip/oelala/src/backend/generation/types.py",
    "/home/flip/oelala/src/backend/app.py",
    "/home/flip/oelala/src/backend/generation/factory.py",
    "/home/flip/oelala/src/backend/generation/lora_utils.py",
    "/home/flip/oelala/src/backend/lora_scanner.py"
]

replacements = {
    "generate-qwen-edit": "generate-cloud-i2i-edit",
    "QwenEditCloudAdapter": "I2IEditCloudAdapter",
    "QWEN_EDIT_MODEL_VARIANTS": "EDIT_MODEL_VARIANTS",
    "DEFAULT_QWEN_MODEL": "DEFAULT_EDIT_MODEL",
    "resolve_qwen_edit_model": "resolve_edit_model",
    "build_qwen_edit_workflow": "build_i2i_edit_workflow",
    "qwen_model": "edit_model",
    "qwen_edit": "i2i_edit",
    "qwen_unet_name": "edit_unet_name",
    "Qwen Image Edit": "I2I Edit",
    "Qwen Edit": "I2I Edit",
    "Qwen instruction-based": "I2I instruction-based",
    "generate_qwen_edit": "generate_cloud_i2i_edit",
    "RUNPOD_QWEN_ENDPOINT_ID": "RUNPOD_I2I_ENDPOINT_ID",
    "runpod_endpoint_qwen": "runpod_endpoint_i2i"
}

keep = ["qwen_image_edit_2511_fp8mixed", "qwen_2.5_vl_7b_fp8", "qwen_image_vae", "TextEncodeQwenImageEditPlus", "qwen_image"]

for fpath in files:
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            content = f.read()
        
        for k, v in replacements.items():
            content = content.replace(k, v)
            
        with open(fpath, "w") as f:
            f.write(content)

