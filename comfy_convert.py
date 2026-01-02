#!/usr/bin/env python3
"""
ComfyUI Workflow to API Format Converter v2.0
Converts standard workflow JSON to API format for terminal execution

This improved version:
1. Uses the node's 'inputs' array to find linked connections
2. Maps widgets_values using the order from ComfyUI's /object_info
3. Handles special nodes like Power Lora Loader and VHS_VideoCombine
"""

import json
import sys
import urllib.request
from pathlib import Path


DEBUG = False  # Set True for verbose output


def log(msg):
    if DEBUG:
        print(f"🔍 {msg}")


def get_node_info_from_comfyui():
    """Fetch node definitions from running ComfyUI"""
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=10)
        return json.loads(req.read().decode())
    except Exception as e:
        print(f"⚠️ Could not fetch node info from ComfyUI: {e}")
        return None


def get_widget_inputs(info):
    """Get list of inputs that become widgets (not connections)"""
    widget_inputs = []
    
    required = info.get('input', {}).get('required', {})
    optional = info.get('input', {}).get('optional', {})
    
    for input_name, input_spec in required.items():
        input_type = input_spec[0] if isinstance(input_spec, list) else input_spec
        # Widget types: dropdown/combo (list), or simple types
        if isinstance(input_type, list) or input_type in ['INT', 'FLOAT', 'STRING', 'BOOLEAN']:
            widget_inputs.append(input_name)
    
    for input_name, input_spec in optional.items():
        input_type = input_spec[0] if isinstance(input_spec, list) else input_spec
        if isinstance(input_type, list) or input_type in ['INT', 'FLOAT', 'STRING', 'BOOLEAN']:
            widget_inputs.append(input_name)
    
    return widget_inputs


def convert_workflow_to_api(workflow_path: str, output_path: str = None):
    """Convert workflow JSON to API format"""
    
    # Load workflow
    with open(workflow_path) as f:
        wf = json.load(f)
    
    # Get node definitions from ComfyUI
    node_info = get_node_info_from_comfyui()
    if not node_info:
        print("❌ ComfyUI must be running to convert workflows")
        return None
    
    # Build link lookup: link_id -> (from_node, from_slot, to_node, to_slot)
    links = {l[0]: {'from': str(l[1]), 'from_slot': l[2], 'to': str(l[3]), 'to_slot': l[4]} 
             for l in wf.get('links', [])}
    
    api_prompt = {}
    issues = []
    
    for node in wf['nodes']:
        node_id = str(node['id'])
        node_type = node['type']
        
        if node_type not in node_info:
            issues.append(f"Unknown node type: {node_type} (node {node_id})")
            continue
        
        info = node_info[node_type]
        inputs = {}
        
        log(f"Processing node {node_id}: {node_type}")
        
        # Step 1: Handle linked inputs from node's inputs array
        linked_inputs = set()
        if 'inputs' in node:
            for inp in node['inputs']:
                if inp.get('link'):
                    link = links.get(inp['link'])
                    if link:
                        inputs[inp['name']] = [link['from'], link['from_slot']]
                        linked_inputs.add(inp['name'])
                        log(f"  Linked: {inp['name']} <- node {link['from']} slot {link['from_slot']}")
        
        # Step 2: Handle widget values
        widgets = node.get('widgets_values', [])
        
        # Special handling for different widget formats
        if isinstance(widgets, dict):
            # VHS_VideoCombine uses dict format directly
            for k, v in widgets.items():
                if k not in linked_inputs and k != 'videopreview':
                    inputs[k] = v
                    log(f"  Widget (dict): {k} = {v}")
        elif isinstance(widgets, list):
            # Standard list format - map to input names
            widget_names = get_widget_inputs(info)
            
            # Filter out linked ones
            widget_names = [n for n in widget_names if n not in linked_inputs]
            
            # Special handling for Power Lora Loader
            if 'Power Lora Loader' in node_type:
                # widgets_values format: [{}, header, lora1, lora2, ..., {}, '']
                lora_idx = 1
                for w in widgets:
                    if isinstance(w, dict) and 'lora' in w and 'on' in w:
                        inputs[f'lora_{lora_idx}'] = w
                        log(f"  LoRA: lora_{lora_idx} = {w.get('lora')}")
                        lora_idx += 1
            else:
                # Standard widget mapping
                widget_idx = 0
                for wname in widget_names:
                    if widget_idx < len(widgets):
                        val = widgets[widget_idx]
                        # Skip None values and special objects
                        if val is not None and not (isinstance(val, dict) and 'type' in val):
                            inputs[wname] = val
                            log(f"  Widget: {wname} = {val}")
                        widget_idx += 1
        
        api_prompt[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }
        
        if 'title' in node:
            api_prompt[node_id]["_meta"] = {"title": node['title']}
    
    # Output path
    if not output_path:
        p = Path(workflow_path)
        output_path = str(p.parent / f"{p.stem}_api.json")
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(api_prompt, f, indent=2)
    
    print(f"✅ Converted: {workflow_path}")
    print(f"   Output: {output_path}")
    print(f"   Nodes: {len(api_prompt)}")
    
    if issues:
        print(f"   ⚠️ Issues: {len(issues)}")
        for issue in issues:
            print(f"      - {issue}")
    
    return output_path


def validate_api_workflow(api_path: str):
    """Validate API workflow by checking with ComfyUI"""
    with open(api_path) as f:
        api = json.load(f)
    
    node_info = get_node_info_from_comfyui()
    if not node_info:
        return False
    
    issues = []
    
    for node_id, node in api.items():
        node_type = node['class_type']
        if node_type not in node_info:
            issues.append(f"Node {node_id}: Unknown type {node_type}")
            continue
        
        info = node_info[node_type]
        required = info.get('input', {}).get('required', {})
        
        # Check required inputs
        for input_name in required:
            if input_name not in node['inputs']:
                # Check if it has a default
                if len(required[input_name]) < 2 or 'default' not in required[input_name][1]:
                    issues.append(f"Node {node_id} ({node_type}): Missing required input '{input_name}'")
    
    if issues:
        print(f"❌ Validation failed with {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print(f"✅ Validation passed for {api_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python comfy_convert.py <workflow.json> [output_api.json]")
        print("       python comfy_convert.py --validate <api.json>")
        print("\nConverts ComfyUI workflow to API format for terminal execution.")
        print("ComfyUI must be running at http://127.0.0.1:8188")
        sys.exit(1)
    
    if sys.argv[1] == '--validate':
        if len(sys.argv) < 3:
            print("Usage: python comfy_convert.py --validate <api.json>")
            sys.exit(1)
        validate_api_workflow(sys.argv[2])
    elif sys.argv[1] == '--debug':
        global DEBUG
        DEBUG = True
        workflow_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        convert_workflow_to_api(workflow_path, output_path)
    else:
        workflow_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        convert_workflow_to_api(workflow_path, output_path)


if __name__ == "__main__":
    main()
