"""
Oelala Storage Node - Local-first distributed media storage.

This module provides:
- Unified media storage with SQLite metadata
- Cross-platform support (Windows/Linux/Mac)
- REST API for node communication
- P2P sync between nodes
"""

from .node import StorageNode
from .models import MediaFile, SyncState, NodeConfig
from .api import create_storage_api

__version__ = "0.1.0"
__all__ = ["StorageNode", "MediaFile", "SyncState", "NodeConfig", "create_storage_api"]
