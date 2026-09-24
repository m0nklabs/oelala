"""
Oelala Storage Node - Local-first distributed media storage.

This module provides:
- Unified media storage with SQLite metadata
- Cross-platform support (Windows/Linux/Mac)
- REST API for node communication
- P2P sync between nodes
"""

from .api import create_storage_api
from .models import MediaFile, NodeConfig, SyncState
from .node import StorageNode

__version__ = "0.1.0"
__all__ = ["MediaFile", "NodeConfig", "StorageNode", "SyncState", "create_storage_api"]
