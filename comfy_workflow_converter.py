#!/usr/bin/env python3
"""
ComfyUI Workflow Converter
Converts standard workflow JSON to API format for terminal execution
"""

import json
import sys
import urllib.request
from pathlib import Path

def get_node_info_from_comfyui():
    """Fetch node definitions from ComfyUI API"""
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=10)
        return json.loads(req.read().decode())
    except Exception as e:
        print(f"⚠️ Could not fetch node info from ComfyUI: {e}")
        return None

def convert_workflow_to_api(workflow_path, output_path=None):
    """Convert standard workflow to API format"""
    
    print(f"📂 Loading workflow: {workflow_path}")
    with open(workflow_path) as f:
        workflow = json.load(f)
    
    # Get node definitions from ComfyUI
    print("📡 Fetching node definitions from ComfyUI...")
    node_info = get_node_info_from_comfyui()
    
    if not node_info:
        print("❌ ComfyUI must be running to convert workflows")
        return None
    
    # Build link map: link_id -> (from_node, from_slot, to_node, to_slot, type)
    link_map = {}
    for link in workflow.get('links', []):
        link_id = link[0]
        link_map[link_id] = {
            'from_node': link[1],
            'from_slot': link[2],
            'to_node': link[3],
            'to_slot': link[4],
            'type': link[5] if len(link) > 5 else None
        }
    
    # Convert each node
    api_workflow = {}
    
    for node in workflow.get('nodes', []):
        node_id = str(node['id'])
        node_type = node['type']
        
        if node_type not in node_info:
            print(f"⚠️ Unknown node type: {node_type} (node {node_id})")
            continue
        
        node_def = node_info[node_type]
        required_inputs = node_def.get('input', {}).get('required', {})
        optional_inputs = node_def.get('input', {}).get('optional', {})
        all_inputs = {**required_inputs, **optional_inputs}
        
        # Build inputs dict
        inputs = {}
        
        # Get widget names in order (non-link inputs)
        widget_names = []
        for input_name, input_def in all_inputs.items():
            # If it's not a connection type, it's a widget
            if isinstance(input_def, list) and len(input_def) > 0:
                input_type = input_def[0]
                # Connection types are usually uppercase like MODEL, CLIP, etc
                if not (isinstance(input_type, str) and input_type.isupper() and len(input_type) > 2):
                    widget_names.append(input_name)
                elif isinstance(input_type, list):
                    # Enum/dropdown widget
                    widget_names.append(input_name)
        
        # Map widget values to input names
        widget_values = node.get('widgets_values', [])
        widget_idx = 0
        
        for input_name, input_def in all_inputs.items():
            if isinstance(input_def, list) and len(input_def) > 0:
                input_type = input_def[0]
                
                # Check if this is a connection type
                is_connection = isinstance(input_type, str) and input_type.isupper() and len(input_type) > 2
                
                if is_connection:
                    # Look for link in node's inputs
                    for node_input in node.get('inputs', []):
                        if node_input.get('name') == input_name and node_input.get('link'):
                            link_info = link_map.get(node_input['link'])
                            if link_info:
                                inputs[input_name] = [str(link_info['from_node']), link_info['from_slot']]
                            break
                else:
                    # It's a widget - get value from widgets_values
                    if widget_idx < len(widget_values):
                        inputs[input_name] = widget_values[widget_idx]
                    widget_idx += 1
        
        api_workflow[node_id] = {
            'class_type': node_type,
            'inputs': inputs
        }
        
        # Add _meta with title if available
        if node.get('title'):
            api_workflow[node_id]['_meta'] = {'title': node['title']}
    
    # Determine output path
    if output_path is None:
        output_path = str(workflow_path).replace('.json', '_api.json')
    
    print(f"💾 Saving API workflow: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(api_workflow, f, indent=2)
    
    print(f"✅ Converted {len(api_workflow)} nodes to API format")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python comfy_workflow_converter.py <workflow.json> [output_api.json]")
        print("\nExample:")
        print("  python comfy_workflow_converter.py 'WAN2.2-I2V-DisTorch2-NEW.json'")
        sys.exit(1)
    
    workflow_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = convert_workflow_to_api(workflow_path, output_path)
    if result:
        print(f"\n🎉 Done! API workflow saved to: {result}")
        print(f"\nTo run via terminal:")
        print(f"  curl -X POST http://127.0.0.1:8188/prompt -H 'Content-Type: application/json' -d @'{result}'")


if __name__ == "__main__":
    main()
