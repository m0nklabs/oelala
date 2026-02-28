"""Oelala SDK exceptions."""

from __future__ import annotations

from typing import Any, Dict, Optional


class OelalaError(Exception):
    """Base exception for all Oelala SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthenticationError(OelalaError):
    """API key is missing, invalid, or expired (HTTP 401)."""
    pass


class RateLimitError(OelalaError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InsufficientCreditsError(OelalaError):
    """Not enough credits for the requested generation (HTTP 402)."""
    pass


class ValidationError(OelalaError):
    """Invalid request parameters (HTTP 422)."""
    pass


class NotFoundError(OelalaError):
    """Resource not found (HTTP 404)."""
    pass


class ServerError(OelalaError):
    """Server-side error (HTTP 5xx)."""
    pass


class TimeoutError(OelalaError):
    """Job polling timed out."""
    pass
