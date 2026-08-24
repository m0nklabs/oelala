"""
Shared utilities for serving objects from MinIO with HTTP Range, ETag,
Last-Modified, and CORS support.

Used by app.py (_storage_proxy_response) and gallery_api.py to avoid
duplicating ~40 lines of Range request parsing logic.
"""

from datetime import datetime
from email.utils import format_datetime
from typing import List, Optional, Tuple

# Single source of truth for CORS allowed origins.
# Imported by app.py (CORSMiddleware + _storage_proxy_response) and
# gallery_api.py (public media endpoint).
ALLOWED_ORIGINS: List[str] = [
    "https://oelala.xyz",
    "http://oelala.xyz",
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:5174",
]


def parse_range_header(
    range_header: str,
    total_size: int,
) -> Optional[Tuple[int, int]]:
    """
    Parse an HTTP Range header value per RFC 7233.

    Supports:
      - ``bytes=0-499``   (explicit range)
      - ``bytes=500-``    (open-ended range)
      - ``bytes=-500``    (suffix byte range — last 500 bytes)

    Args:
        range_header: The raw Range header value (e.g. ``"bytes=0-499"``).
        total_size: Total size of the object in bytes.

    Returns:
        ``(range_start, range_end)`` tuple on success, or ``None`` if the
        header is malformed or unsatisfiable.

    Raises:
        ValueError: If the range is syntactically valid but unsatisfiable
            (e.g. start > end, start >= total_size, suffix <= 0).
            Callers should return 416 Range Not Satisfiable for these.
    """
    if not range_header or not range_header.startswith("bytes="):
        return None

    range_spec = range_header[6:].strip()
    if "," in range_spec:
        raise ValueError("Multiple byte ranges are not supported")

    parts = range_spec.split("-", 1)
    if len(parts) != 2:
        return None  # malformed

    start_str, end_str = parts[0].strip(), parts[1].strip()

    if not start_str:
        # Suffix byte range: bytes=-N means the last N bytes
        suffix_length = int(end_str)
        if suffix_length <= 0:
            raise ValueError("Suffix byte range must be greater than zero")
        suffix_length = min(suffix_length, total_size)
        range_start = max(total_size - suffix_length, 0)
        range_end = total_size - 1
    else:
        range_start = int(start_str)
        if range_start < 0:
            raise ValueError("Range start must not be negative")
        range_end = int(end_str) if end_str else total_size - 1
        if range_end < 0:
            raise ValueError("Range end must not be negative")
        range_end = min(range_end, total_size - 1)

    if range_start > range_end or range_start >= total_size:
        raise ValueError("Range not satisfiable")

    return range_start, range_end


def format_last_modified(dt: datetime) -> str:
    """Format a datetime for the HTTP Last-Modified header."""
    return format_datetime(dt, usegmt=True)
