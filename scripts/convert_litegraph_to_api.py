#!/usr/bin/env python3
"""
Convert a ComfyUI litegraph-format workflow to API format.
Uses the ComfyUI object_info endpoint to correctly map widget_values to input names.
"""

import json
import sys
import requests
from pathlib import Path

COMFYUI_URL = "http://localhost:8188"


def get_object_info():
    """Fetch all node type definitions from ComfyUI."""
    resp = requests.get(f"{COMFYUI_URL}/object_info")
    resp.raise_for_status()
    return resp.json()


def get_widget_inputs(node_def: dict) -> list[str]:
    """
    Extract the ordered list of widget input names from a node definition.
    Widget inputs are those in 'required' or 'optional' that are NOT connection types.
    Connection types are lists starting with a string type name (like ['MODEL'], ['IMAGE']).
    Widget types are lists starting with a type like 'INT', 'FLOAT', 'STRING', 'BOOLEAN',
    or a list of choices.
    """
    widget_names = []
    
    for section in ['required', 'optional']:
        inputs = node_def.get('input', {}).get(section, {})
        for name, spec in inputs.items():
            if not isinstance(spec, (list, tuple)) or len(spec) == 0:
                continue
            
            type_info = spec[0]
            
            # Connection types: single string that's an uppercase type name
            # Widget types: 'INT', 'FLOAT', 'STRING', 'BOOLEAN', list of choices, 'COMBO'
            if isinstance(type_info, str):
                # Known widget types
                if type_info in ('INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'):
                    widget_names.append(name)
                elif type_info.startswith('*') or type_info == 'UNIQUE_ID':
                    widget_names.append(name)
                # Connection types are typically uppercase like 'MODEL', 'IMAGE', 'LATENT', etc.
                # But some widget types are also uppercase like 'STRING', 'INT'
                # The key difference: connection types don't have default values in spec[1]
                # and widget types usually do
                elif len(spec) > 1 and isinstance(spec[1], dict):
                    # Has config dict - likely a widget
                    widget_names.append(name)
                # else: connection type, skip
            elif isinstance(type_info, list):
                # List of choices = combo widget
                widget_names.append(name)
    
    return widget_names


def convert_workflow(litegraph_path: str, output_path: str = None):
    """Convert litegraph workflow to API format."""
    
    # Load litegraph workflow
    with open(litegraph_path) as f:
        wf = json.load(f)
    
    nodes = wf.get('nodes', [])
    links = wf.get('links', [])
    
    # Build link map: link_id -> (src_node_id, src_slot)
    link_map = {}
    for link in links:
        link_id, src_node, src_slot, dst_node, dst_slot = link[0], link[1], link[2], link[3], link[4]
        link_map[link_id] = (src_node, src_slot)
    
    # Get node definitions from ComfyUI
    print("Fetching node definitions from ComfyUI...")
    object_info = get_object_info()
    
    # Build API workflow
    api_workflow = {}
    skipped = []
    
    for node in nodes:
        node_id = str(node['id'])
        class_type = node.get('type', '')
        
        # Skip UI-only nodes
        if class_type in ('Note', 'Reroute', 'PrimitiveNode'):
            skipped.append(f"{node_id} ({class_type})")
            continue
        
        # Check if node is muted/bypassed
        mode = node.get('mode', 0)
        # mode 0 = active, 2 = muted, 4 = bypassed
        
        inputs = {}
        
        # 1. Map connected inputs
        connected_input_names = set()
        if node.get('inputs'):
            for inp in node['inputs']:
                inp_name = inp['name']
                link_id = inp.get('link')
                if link_id is not None and link_id in link_map:
                    src_node_id, src_slot = link_map[link_id]
                    inputs[inp_name] = [str(src_node_id), src_slot]
                    connected_input_names.add(inp_name)
        
        # 2. Map widget values using object_info
        widget_values = node.get('widgets_values', [])
        if widget_values and isinstance(widget_values, dict):
            # Dict format (e.g., VHS_VideoCombine) - keys are already input names
            for wname, val in widget_values.items():
                if wname not in connected_input_names and val is not None:
                    inputs[wname] = val
        elif widget_values and isinstance(widget_values, list) and class_type in object_info:
            node_def = object_info[class_type]
            widget_names = get_widget_inputs(node_def)
            
            # Filter out already-connected inputs
            available_widgets = [w for w in widget_names if w not in connected_input_names]
            
            # Map values to names
            val_idx = 0
            for wname in available_widgets:
                if val_idx >= len(widget_values):
                    break
                val = widget_values[val_idx]
                # Skip None values that might be control widgets
                if val is not None:
                    inputs[wname] = val
                val_idx += 1
        elif widget_values and class_type not in object_info:
            print(f"  ⚠️ Node {node_id} ({class_type}): not in object_info, storing raw widget_values")
            inputs['_raw_widget_values'] = widget_values
        
        api_workflow[node_id] = {
            'class_type': class_type,
            'inputs': inputs,
        }
        
        # Add metadata for reference
        title = node.get('title', '')
        if title:
            api_workflow[node_id]['_meta'] = {'title': title}
    
    print(f"\n✅ Converted {len(api_workflow)} nodes ({len(skipped)} skipped)")
    
    # Determine output path
    if output_path is None:
        p = Path(litegraph_path)
        output_path = str(p.parent / (p.stem + '-api' + p.suffix))
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(api_workflow, f, indent=2)
    
    print(f"📁 Saved to: {output_path}")
    return api_workflow


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: convert_litegraph_to_api.py <input.json> [output.json]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_workflow(input_path, output_path)
