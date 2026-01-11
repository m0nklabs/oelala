"""
Tests for API key authentication module.
"""

import pytest
import hashlib
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src/backend to path
backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))


def test_generate_api_key():
    """Test API key generation."""
    from api_key_auth import generate_api_key

    full_key, key_hash, key_prefix = generate_api_key()

    # Check format
    assert full_key.startswith("oelala_")
    assert len(full_key) == 71  # "oelala_" (7) + 64 hex chars

    # Check hash
    expected_hash = hashlib.sha256(full_key.encode()).hexdigest()
    assert key_hash == expected_hash

    # Check prefix
    assert key_prefix == full_key[:15]
    assert key_prefix.startswith("oelala_")


def test_hash_api_key():
    """Test API key hashing."""
    from api_key_auth import hash_api_key

    key = "oelala_test_key_12345"
    hash1 = hash_api_key(key)
    hash2 = hash_api_key(key)

    # Same key should produce same hash
    assert hash1 == hash2

    # Hash should be SHA-256 (64 hex chars)
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Different keys should produce different hashes
    hash3 = hash_api_key("oelala_different_key")
    assert hash3 != hash1


def test_generate_api_key_uniqueness():
    """Test that generated API keys are unique."""
    from api_key_auth import generate_api_key

    keys = [generate_api_key() for _ in range(100)]

    # All full keys should be unique
    full_keys = [k[0] for k in keys]
    assert len(full_keys) == len(set(full_keys))

    # All hashes should be unique
    hashes = [k[1] for k in keys]
    assert len(hashes) == len(set(hashes))


@pytest.mark.asyncio
async def test_validate_api_key_db_success():
    """Test successful API key validation."""
    from api_key_auth import validate_api_key_db, hash_api_key

    test_key = "oelala_test_key"
    test_hash = hash_api_key(test_key)

    # Mock Supabase response
    with patch("credits.get_credit_manager") as mock_manager:
        mock_supabase = Mock()
        mock_result = Mock()
        mock_result.data = [
            {
                "valid": True,
                "user_id": "user-123",
                "key_id": "key-456",
                "error": None,
            }
        ]
        mock_result.execute = lambda: mock_result
        mock_supabase.rpc = Mock(return_value=mock_result)

        mock_manager.return_value.supabase = mock_supabase
        mock_manager.return_value.service_key = "test-key"

        result = await validate_api_key_db(test_hash)

        assert result is not None
        user_id, key_id = result
        assert user_id == "user-123"
        assert key_id == "key-456"


@pytest.mark.asyncio
async def test_validate_api_key_db_invalid():
    """Test validation with invalid API key."""
    from api_key_auth import validate_api_key_db, hash_api_key

    test_hash = hash_api_key("oelala_invalid_key")

    # Mock Supabase response for invalid key
    with patch("credits.get_credit_manager") as mock_manager:
        mock_supabase = Mock()
        mock_result = Mock()
        mock_result.data = [
            {
                "valid": False,
                "user_id": None,
                "key_id": None,
                "error": "Invalid API key",
            }
        ]
        mock_result.execute = lambda: mock_result
        mock_supabase.rpc = Mock(return_value=mock_result)

        mock_manager.return_value.supabase = mock_supabase
        mock_manager.return_value.service_key = "test-key"

        result = await validate_api_key_db(test_hash)

        assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_db_no_service_key():
    """Test validation when Supabase is not configured."""
    from api_key_auth import validate_api_key_db, hash_api_key

    test_hash = hash_api_key("oelala_test_key")

    # Mock with no service key
    with patch("credits.get_credit_manager") as mock_manager:
        mock_manager.return_value.service_key = None

        result = await validate_api_key_db(test_hash)

        assert result is None


@pytest.mark.asyncio
async def test_get_api_key_user_success():
    """Test successful user extraction from API key."""
    from api_key_auth import get_api_key_user

    # Mock validate_api_key_db
    with patch("api_key_auth.validate_api_key_db") as mock_validate:
        mock_validate.return_value = ("user-123", "key-456")

        user = await get_api_key_user("oelala_test_key")

        assert user.id == "user-123"
        assert user.role == "authenticated"
        assert user.app_metadata["api_key_id"] == "key-456"


@pytest.mark.asyncio
async def test_get_api_key_user_missing_key():
    """Test that missing API key raises 401."""
    from api_key_auth import get_api_key_user
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await get_api_key_user(None)

    assert exc_info.value.status_code == 401
    assert "API key required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_api_key_user_invalid_format():
    """Test that invalid format raises 401."""
    from api_key_auth import get_api_key_user
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await get_api_key_user("invalid_format")

    assert exc_info.value.status_code == 401
    assert "Invalid API key format" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_api_key_user_invalid_key():
    """Test that invalid API key raises 401."""
    from api_key_auth import get_api_key_user
    from fastapi import HTTPException

    # Mock validate_api_key_db to return None
    with patch("api_key_auth.validate_api_key_db") as mock_validate:
        mock_validate.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_api_key_user("oelala_invalid_key")

        assert exc_info.value.status_code == 401
        assert "Invalid or expired" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_optional_api_key_user_with_key():
    """Test optional user extraction with valid key."""
    from api_key_auth import get_optional_api_key_user

    # Mock validate_api_key_db
    with patch("api_key_auth.validate_api_key_db") as mock_validate:
        mock_validate.return_value = ("user-123", "key-456")

        user = await get_optional_api_key_user("oelala_test_key")

        assert user is not None
        assert user.id == "user-123"


@pytest.mark.asyncio
async def test_get_optional_api_key_user_without_key():
    """Test optional user extraction with no key."""
    from api_key_auth import get_optional_api_key_user

    user = await get_optional_api_key_user(None)

    assert user is None


@pytest.mark.asyncio
async def test_get_optional_api_key_user_invalid_key():
    """Test optional user extraction with invalid key returns None."""
    from api_key_auth import get_optional_api_key_user

    # Mock validate_api_key_db to return None
    with patch("api_key_auth.validate_api_key_db") as mock_validate:
        mock_validate.return_value = None

        user = await get_optional_api_key_user("oelala_invalid_key")

        assert user is None
