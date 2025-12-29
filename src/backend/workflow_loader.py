#!/usr/bin/env python3
"""
ComfyUI Workflow Loader
Loads workflow templates from JSON files with configurable parameters
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Default workflow directory
WORKFLOW_DIR = Path(__file__).parent.parent.parent / "workflows"


class WorkflowConfig:
    """Configuration for a workflow with UI-exposed parameters"""
    
    def __init__(
        self,
        name: str,
        file_path: Path,
        category: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]]
    ):
        self.name = name
        self.file_path = file_path
        self.category = category
        self.description = description
        self.parameters = parameters  # {param_name: {node_id, input_key, type, default, min, max, options, label}}
        self._template: Optional[Dict] = None
    
    @property
    def template(self) -> Dict:
        """Lazy load the workflow template"""
        if self._template is None:
            self._template = load_workflow_json(self.file_path)
        return self._template
    
    def build(self, **kwargs) -> Dict:
        """Build a workflow with custom parameters"""
        workflow = copy.deepcopy(self.template)
        
        for param_name, value in kwargs.items():
            if param_name in self.parameters:
                param_config = self.parameters[param_name]
                node_id = param_config["node_id"]
                input_key = param_config["input_key"]
                
                if node_id in workflow and "inputs" in workflow[node_id]:
                    workflow[node_id]["inputs"][input_key] = value
                    logger.debug(f"Set {param_name}: node {node_id}.{input_key} = {value}")
        
        return workflow
    
    def to_dict(self) -> Dict:
        """Export config for frontend"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": {
                name: {
                    "type": cfg.get("type", "string"),
                    "default": cfg.get("default"),
                    "min": cfg.get("min"),
                    "max": cfg.get("max"),
                    "step": cfg.get("step"),
                    "options": cfg.get("options"),
                    "label": cfg.get("label", name),
                    "description": cfg.get("description", ""),
                }
                for name, cfg in self.parameters.items()
            }
        }


def load_workflow_json(file_path: Path) -> Dict:
    """Load a workflow JSON file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Workflow not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_api_format(workflow: Dict) -> Dict:
    """
    Convert ComfyUI UI workflow format to API format.
    
    UI format has: {"nodes": [...], "links": [...], "groups": [...]}
    API format has: {"node_id": {"class_type": "...", "inputs": {...}}, ...}
    """
    # Check if already in API format
    if "nodes" not in workflow:
        # Already API format or simple dict
        return workflow
    
    api_workflow = {}
    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])
    
    # Build link map: link_id -> (source_node_id, source_slot)
    link_map = {}
    for link in links:
        # link format: [link_id, source_node_id, source_slot, target_node_id, target_slot, type]
        if len(link) >= 5:
            link_id = link[0]
            source_node_id = link[1]
            source_slot = link[2]
            link_map[link_id] = (str(source_node_id), source_slot)
    
    for node in nodes:
        node_id = str(node.get("id"))
        class_type = node.get("type")
        
        if not class_type:
            continue
        
        inputs = {}
        
        # Process widget values (direct inputs)
        widgets_values = node.get("widgets_values", [])
        widget_inputs = node.get("widget_inputs", {})
        
        # Get input definitions from the node
        node_inputs = node.get("inputs", [])
        
        # Map widget values to input names
        # This is tricky because widget order matters
        widget_idx = 0
        for inp in node_inputs:
            inp_name = inp.get("name")
            inp_link = inp.get("link")
            
            if inp_link is not None:
                # This input is connected to another node
                if inp_link in link_map:
                    source_node, source_slot = link_map[inp_link]
                    inputs[inp_name] = [source_node, source_slot]
            elif widget_idx < len(widgets_values):
                # This might be a widget value
                inputs[inp_name] = widgets_values[widget_idx]
                widget_idx += 1
        
        # Also check for properties that might be inputs
        if "properties" in node:
            for prop_name, prop_value in node["properties"].items():
                if prop_name not in inputs:
                    inputs[prop_name] = prop_value
        
        api_workflow[node_id] = {
            "class_type": class_type,
            "inputs": inputs
        }
    
    return api_workflow


class WorkflowRegistry:
    """Registry of available workflows with their configurations"""
    
    def __init__(self, workflow_dir: Path = WORKFLOW_DIR):
        self.workflow_dir = workflow_dir
        self.workflows: Dict[str, WorkflowConfig] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load workflow configurations from registry file"""
        registry_file = self.workflow_dir / "registry.json"
        
        if registry_file.exists():
            with open(registry_file, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
            
            for workflow_id, config in registry_data.get("workflows", {}).items():
                file_path = self.workflow_dir / config["file"]
                
                self.workflows[workflow_id] = WorkflowConfig(
                    name=config.get("name", workflow_id),
                    file_path=file_path,
                    category=config.get("category", "Other"),
                    description=config.get("description", ""),
                    parameters=config.get("parameters", {})
                )
        else:
            logger.warning(f"No registry.json found at {registry_file}")
    
    def get(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Get a workflow config by ID"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self, category: Optional[str] = None) -> List[Dict]:
        """List available workflows, optionally filtered by category"""
        result = []
        for wf_id, config in self.workflows.items():
            if category is None or config.category == category:
                info = config.to_dict()
                info["id"] = wf_id
                result.append(info)
        return result
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories"""
        return list(set(wf.category for wf in self.workflows.values()))
    
    def build_workflow(self, workflow_id: str, **params) -> Dict:
        """Build a workflow with parameters"""
        config = self.get(workflow_id)
        if not config:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        return config.build(**params)


# Global registry instance
_registry: Optional[WorkflowRegistry] = None


def get_registry() -> WorkflowRegistry:
    """Get the global workflow registry"""
    global _registry
    if _registry is None:
        _registry = WorkflowRegistry()
    return _registry


def reload_registry():
    """Reload the workflow registry"""
    global _registry
    _registry = WorkflowRegistry()
    return _registry
