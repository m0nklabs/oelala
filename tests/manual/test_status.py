import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

from fastapi.testclient import TestClient
from app import app
from auth import get_current_user, User

def override_get_current_user():
    return User(
        id="97833cbb-ed5b-40f9-ab32-033877dcf77d",
        email="mark.op.mobiel@gmail.com",
        role="admin",
        tier="pro",
        credits=999,
        created_at="2026-01-01"
    )

app.dependency_overrides[get_current_user] = override_get_current_user
app.user_middleware.clear()

with TestClient(app) as client:
    response = client.get("/runpod/job/bcda7dd8-37cf-4ed7-9ef1-957f29b915d3-e1")
    print(response.status_code)
    print(response.json())
