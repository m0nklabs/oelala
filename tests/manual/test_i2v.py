import asyncio
import os
import sys

# Change working directory so imports work
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import get_supabase_client
from generation.v1_compat import dispatch_v1
from generation.types import Operation, MediaType
from schemas import User

async def main():
    supabase = get_supabase_client()
    # Hack to get a user
    res = supabase.table("users").select("*").eq("email", "mark.op.mobiel@gmail.com").limit(1).execute()
    users = res.data
    if not users:
        res = supabase.table("users").select("*").limit(1).execute()
        users = res.data
    if not users:
        print("No users found")
        return

    u = dict(users[0])
    print(f"Using user: {u['email']}")

    # Needs to match Pydantic schema for User
    user_obj = User(
        id=u["id"],
        email=u["email"],
        role=u.get("role", "admin"),
        tier=u.get("tier", "pro"),
        credits=int(u.get("credits", 999)),
        created_at=str(u.get("created_at", ""))
    )

    file_path = "/home/flip/oelala/uploads/cat_ak.png"
    class DummyFile:
        def __init__(self):
            self.filename = "cat_ak.png"
            self.content_type = "image/png"
            self._file = open(file_path, "rb")
        async def read(self):
            self._file.seek(0)
            return self._file.read()

    res = await dispatch_v1(
        form=dict(
            prompt="The camera intensely shakes as the enraged cat fires the golden AK-47 in full auto, bright muzzle flashes illuminating its fur, the massive fiery explosion expanding and billowing in the cinematic background, slow motion, hyperrealistic.",
            negative_prompt="low quality, blurry, distorted, artifacts, flickering, jitter",
            mode="i2v",
            num_frames=81,
            resolution="720p",
            fps=16,
            aspect_ratio="5:4",
            steps=15,
            cfg=3.0,
            seed=-1,
            high_noise_steps=8,
            shift=8.0,
            sampler_name="dpmpp_2m",
            scheduler="beta",
            lora_configs="",
        ),
        files={"file": DummyFile()},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-cloud-i2v",
        user=user_obj,
        register_job_settings={
            "job_type": "i2v",
        },
        v1_format="cloud",
    )
    print("Job submitted! Result:", res)

if __name__ == "__main__":
    asyncio.run(main())
