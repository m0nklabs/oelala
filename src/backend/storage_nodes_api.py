import os
import logging
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from supabase import create_client, Client

# Use centralized Supabase client if possible, else create one
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/storage-nodes", tags=["Storage Nodes"])

class HeartbeatPayload(BaseModel):
    node_id: str
    name: str
    type: str
    total_bytes: int
    used_bytes: int
    status: str
    version: str

def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Missing Supabase config")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def verify_storage_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
        
    token = parts[1]
    expected_token = os.environ.get("STORAGE_NODE_API_KEY", "dev-secret-key-12345")
    
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid storage node API key")
        
    return True

@router.post("/heartbeat")
async def receive_heartbeat(
    payload: HeartbeatPayload, 
    request: Request,
    _: bool = Depends(verify_storage_api_key)
):
    """
    Receive heartbeat from a distributed oelala-storage node.
    Register it if new, update stats & timestamp if existing.
    """
    client = get_supabase()
    
    # Try to get client IP for debugging
    client_ip = None
    if request.client:
        client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    try:
        # Upsert the storage node heartbeat
        data = {
            "node_id": payload.node_id,
            "name": payload.name,
            "type": payload.type,
            "total_bytes": payload.total_bytes,
            "used_bytes": payload.used_bytes,
            "status": payload.status,
            "version": payload.version,
            "ip_address": client_ip,
            "last_heartbeat_at": datetime.now(timezone.utc).isoformat()
        }
        
        # We need to do an upsert on node_id.
        # Ensure node_id is unique constraint in DB for this to work natively.
        response = client.table("storage_nodes").upsert(
            data, 
            on_conflict="node_id"
        ).execute()
        
        return {"status": "success", "received": True}
        
    except Exception as e:
        logger.error(f"Failed to process storage node heartbeat: {e}")
        raise HTTPException(status_code=500, detail="Database failure")

@router.get("")
async def list_storage_nodes(client: Client = Depends(get_supabase)):
    """List all registered storage nodes, for the admin dashboard"""
    try:
        response = client.table("storage_nodes").select("*").order("name").execute()
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch storage nodes: {e}")
        raise HTTPException(status_code=500, detail="Database failure")
