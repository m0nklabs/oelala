import os
import sys
import asyncio

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

# Disable cors validation failures for test client
app.user_middleware.clear()

client = TestClient(app)

file_path = "/home/flip/oelala/uploads/cat_ak.png"

with TestClient(app) as client:
    # TestClient implicitly fires startup events, initializing V2 engine
    with open(file_path, "rb") as f:
        response = client.post(
            "/generate-cloud-wan22-async",
            data={
                "prompt": "The camera intensely shakes as the enraged cat fires the golden AK-47 in full auto, bright muzzle flashes illuminating its fur, the massive fiery explosion expanding and billowing in the cinematic background, slow motion.",
                "negative_prompt": "low quality, blurry, distorted, artifacts, flickering, jitter, stationary",
                "mode": "i2v",
                "num_frames": "41",
                "resolution": "480p",
                "fps": "16",
                "aspect_ratio": "5:4",
            },
            files={"file": ("cat_ak.png", f, "image/png")}
        )

print(response.status_code)
print(response.json())
