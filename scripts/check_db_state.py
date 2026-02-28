#!/usr/bin/env python3
"""Check Supabase DB state for gallery likes/views."""
import os
import sys
sys.path.insert(0, '/home/flip/oelala')

from dotenv import load_dotenv
load_dotenv('/home/flip/oelala/.env')

from supabase import create_client

sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])

# Check published_media columns
try:
    r = sb.table('published_media').select('id,view_count,like_count').limit(1).execute()
    print('✅ published_media has view_count/like_count:', r.data)
except Exception as e:
    print('❌ published_media error:', e)

# Check published_media_likes table
try:
    r = sb.table('published_media_likes').select('id').limit(1).execute()
    print('✅ published_media_likes exists:', r.data)
except Exception as e:
    print('❌ published_media_likes missing:', e)

# Check toggle_like RPC
try:
    r = sb.rpc('toggle_like', {'p_media_id': '00000000-0000-0000-0000-000000000000'}).execute()
    print('✅ toggle_like RPC exists:', r.data)
except Exception as e:
    print('❌ toggle_like RPC missing:', e)

# Check increment_view_count RPC
try:
    r = sb.rpc('increment_view_count', {'p_media_id': '00000000-0000-0000-0000-000000000000'}).execute()
    print('✅ increment_view_count RPC exists:', r.data)
except Exception as e:
    print('❌ increment_view_count RPC missing:', e)
