#!/usr/bin/env python3
"""
Profile API Integration Test
Tests user profile creation, retrieval, and updates
"""

import pytest
import httpx
import os
from datetime import datetime

# Test configuration
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7998")
TEST_TOKEN = os.getenv("TEST_JWT_TOKEN", "")  # Valid Supabase JWT token for testing

# Skip if no token provided
pytestmark = pytest.mark.skipif(
    not TEST_TOKEN, reason="TEST_JWT_TOKEN not set - skipping profile API tests"
)


@pytest.fixture
def auth_headers():
    """Headers with authentication"""
    return {
        "Authorization": f"Bearer {TEST_TOKEN}",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_get_my_profile(auth_headers):
    """Test retrieving authenticated user's profile"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/profile/me",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert "username" in data
        assert "display_name" in data
        assert "is_public" in data
        assert "created_at" in data

        print(f"✅ Profile retrieved: {data['username']}")


@pytest.mark.asyncio
async def test_update_my_profile(auth_headers):
    """Test updating authenticated user's profile"""
    async with httpx.AsyncClient() as client:
        # Update profile
        update_data = {
            "display_name": "Test User Updated",
            "bio": "This is a test bio for integration testing",
            "is_public": True,
        }

        response = await client.put(
            f"{BASE_URL}/api/profile/me",
            headers=auth_headers,
            json=update_data,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify updates
        assert data["display_name"] == update_data["display_name"]
        assert data["bio"] == update_data["bio"]
        assert data["is_public"] == update_data["is_public"]

        print(f"✅ Profile updated: {data['display_name']}")


@pytest.mark.asyncio
async def test_get_profile_by_username(auth_headers):
    """Test retrieving profile by username"""
    async with httpx.AsyncClient() as client:
        # First get own profile to get username
        me_response = await client.get(
            f"{BASE_URL}/api/profile/me",
            headers=auth_headers,
        )
        assert me_response.status_code == 200
        username = me_response.json()["username"]

        # Now get profile by username (no auth required for public profiles)
        response = await client.get(
            f"{BASE_URL}/api/profile/username/{username}",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == username

        print(f"✅ Profile retrieved by username: {username}")


@pytest.mark.asyncio
async def test_get_profile_stats(auth_headers):
    """Test retrieving user statistics"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/profile/me/stats",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify stats structure
        assert "total_media" in data
        assert "published_media" in data
        assert "total_likes_received" in data
        assert "total_views_received" in data

        # All should be non-negative integers
        assert data["total_media"] >= 0
        assert data["published_media"] >= 0
        assert data["total_likes_received"] >= 0
        assert data["total_views_received"] >= 0

        print(f"✅ Stats retrieved: {data['total_media']} media, {data['published_media']} published")


@pytest.mark.asyncio
async def test_update_username(auth_headers):
    """Test updating username"""
    async with httpx.AsyncClient() as client:
        # Generate unique username with timestamp
        timestamp = int(datetime.now().timestamp())
        new_username = f"testuser_{timestamp}"

        response = await client.put(
            f"{BASE_URL}/api/profile/me",
            headers=auth_headers,
            json={"username": new_username},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == new_username.lower()  # Usernames stored in lowercase

        print(f"✅ Username updated to: {new_username}")


@pytest.mark.asyncio
async def test_invalid_username_format(auth_headers):
    """Test that invalid username format is rejected"""
    async with httpx.AsyncClient() as client:
        # Test with special characters
        response = await client.put(
            f"{BASE_URL}/api/profile/me",
            headers=auth_headers,
            json={"username": "invalid@username!"},
        )

        assert response.status_code == 422  # Validation error

        print("✅ Invalid username correctly rejected")


@pytest.mark.asyncio
async def test_profile_without_auth():
    """Test that profile endpoints require authentication"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/profile/me")

        # Should return 401 Unauthorized
        assert response.status_code == 401

        print("✅ Unauthenticated request correctly rejected")


@pytest.mark.asyncio
async def test_social_links_update(auth_headers):
    """Test updating social media links"""
    async with httpx.AsyncClient() as client:
        social_links = {
            "twitter": "https://twitter.com/testuser",
            "github": "https://github.com/testuser",
            "website": "https://example.com",
        }

        response = await client.put(
            f"{BASE_URL}/api/profile/me",
            headers=auth_headers,
            json={"social_links": social_links},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["social_links"] == social_links

        print(f"✅ Social links updated: {len(social_links)} links")


if __name__ == "__main__":
    print("🧪 Running Profile API Integration Tests")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Auth Token: {'✅ Set' if TEST_TOKEN else '❌ Not Set'}")
    print("=" * 60)

    # Run tests
    pytest.main([__file__, "-v", "-s"])
