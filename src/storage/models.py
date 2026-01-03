"""
Storage Node data models using SQLite with SQLAlchemy.
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid


class MediaType(str, Enum):
    """Type of media file."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OTHER = "other"


class SyncStatus(str, Enum):
    """Synchronization status for files."""
    LOCAL_ONLY = "local_only"      # Only exists on this node
    SYNCED = "synced"              # Synced with peers
    PENDING_UPLOAD = "pending_upload"  # Needs to sync to peers
    PENDING_DOWNLOAD = "pending_download"  # Needs to download from peer
    CONFLICT = "conflict"          # Conflict detected


class NodeType(str, Enum):
    """Type of storage node."""
    PRIMARY = "primary"    # Main node, always online
    REPLICA = "replica"    # Mirror/failover
    EDGE = "edge"          # Local cache, partial sync
    ARCHIVE = "archive"    # Cold storage


@dataclass
class MediaFile:
    """Represents a media file in the storage system."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    original_filename: str = ""
    file_path: str = ""  # Relative to media root
    content_hash: str = ""  # SHA-256 hash
    size_bytes: int = 0
    media_type: MediaType = MediaType.OTHER
    mime_type: str = ""
    
    # Metadata
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    
    # Generation info (if AI-generated)
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    workflow_json: Optional[str] = None
    
    # User info (for multi-tenant)
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    
    # Sync info
    sync_status: SyncStatus = SyncStatus.LOCAL_ONLY
    last_synced_at: Optional[datetime] = None
    source_node_id: Optional[str] = None
    
    # Tags and organization
    tags: List[str] = field(default_factory=list)
    favorite: bool = False
    archived: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type.value,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model": self.model,
            "seed": self.seed,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "sync_status": self.sync_status.value,
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "source_node_id": self.source_node_id,
            "tags": self.tags,
            "favorite": self.favorite,
            "archived": self.archived,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MediaFile":
        """Create from dictionary."""
        data = data.copy()
        if "media_type" in data:
            data["media_type"] = MediaType(data["media_type"])
        if "sync_status" in data:
            data["sync_status"] = SyncStatus(data["sync_status"])
        for field in ["created_at", "modified_at", "last_synced_at"]:
            if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


@dataclass
class SyncState:
    """Tracks sync state between nodes."""
    
    node_id: str
    peer_node_id: str
    last_sync_at: Optional[datetime] = None
    files_synced: int = 0
    files_pending: int = 0
    bytes_transferred: int = 0
    sync_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "peer_node_id": self.peer_node_id,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "files_synced": self.files_synced,
            "files_pending": self.files_pending,
            "bytes_transferred": self.bytes_transferred,
            "sync_errors": self.sync_errors,
        }


@dataclass
class NodeConfig:
    """Configuration for a storage node."""
    
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.PRIMARY
    node_name: str = "Oelala Node"
    
    # Storage paths
    media_root: str = "/home/flip/oelala/media"
    uploads_dir: str = "uploads"
    generated_dir: str = "generated"
    temp_dir: str = "temp"
    archive_dir: str = "archive"
    
    # Limits
    max_storage_gb: float = 100.0
    max_file_size_mb: float = 500.0
    
    # Network
    api_host: str = "0.0.0.0"
    api_port: int = 7999
    
    # Sync settings
    sync_enabled: bool = True
    sync_interval_seconds: int = 300  # 5 minutes
    sync_peers: List[dict] = field(default_factory=list)
    
    # Security
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    
    # Compression
    compression_enabled: bool = False
    compression_algorithm: str = "lz4"  # lz4, zstd, gzip
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "node_name": self.node_name,
            "media_root": self.media_root,
            "uploads_dir": self.uploads_dir,
            "generated_dir": self.generated_dir,
            "temp_dir": self.temp_dir,
            "archive_dir": self.archive_dir,
            "max_storage_gb": self.max_storage_gb,
            "max_file_size_mb": self.max_file_size_mb,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "sync_enabled": self.sync_enabled,
            "sync_interval_seconds": self.sync_interval_seconds,
            "sync_peers": self.sync_peers,
            "encryption_enabled": self.encryption_enabled,
            "compression_enabled": self.compression_enabled,
            "compression_algorithm": self.compression_algorithm,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NodeConfig":
        data = data.copy()
        if "node_type" in data:
            data["node_type"] = NodeType(data["node_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def load(cls, config_path: str) -> "NodeConfig":
        """Load config from JSON file."""
        with open(config_path, "r") as f:
            return cls.from_dict(json.load(f))
    
    def save(self, config_path: str):
        """Save config to JSON file."""
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @property
    def uploads_path(self) -> Path:
        return Path(self.media_root) / self.uploads_dir
    
    @property
    def generated_path(self) -> Path:
        return Path(self.media_root) / self.generated_dir
    
    @property
    def temp_path(self) -> Path:
        return Path(self.media_root) / self.temp_dir
    
    @property
    def archive_path(self) -> Path:
        return Path(self.media_root) / self.archive_dir


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for content addressing."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def detect_media_type(filename: str) -> MediaType:
    """Detect media type from filename extension."""
    ext = Path(filename).suffix.lower()
    
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
    video_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4v"}
    audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
    
    if ext in image_exts:
        return MediaType.IMAGE
    elif ext in video_exts:
        return MediaType.VIDEO
    elif ext in audio_exts:
        return MediaType.AUDIO
    else:
        return MediaType.OTHER


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"
