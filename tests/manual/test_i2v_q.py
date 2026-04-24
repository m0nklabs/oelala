import asyncio
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generation.v1_compat import dispatch_v1
from generation.types import Operation, MediaType
from schemas import User

async def main():
    u = User(
        id="97833cbb-ed5b-40f9-ab32-033877dcf77d",  # the real user id
        email="mark.op.mobiel@gmail.com",
        role="admin",
        tier="pro",
        credits=999,
        created_at="2026-01-01"
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
    
    import generation.v1_compat
    from generation.v2_api import GenerationRouter
    # Setup V2 router structure
    generation.v1_compat._router = GenerationRouter()
    
    # We must patch the router's generation logic since it wasn't started inside the app correctly
    from generation.factory import _registry
    if not _registry.adapters:
        from generation.factory import setup_v2_engine
        await setup_v2_engine()
        
    res = await dispatch_v1(
        form=dict(
            prompt="The camera intensely shakes as the enraged cat fires the golden AK-47 in full auto, bright muzzle flashes illuminating its fur, the massive fiery explosion expanding and billowing in the cinematic background, slow motion.",
            negative_prompt="low quality, blurry, distorted, artifacts, flickering, jitter, stationary",
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
        user=u,
        register_job_settings={
            "job_type": "i2v",
        },
        v1_format="cloud",
    )
    print("Job submitted! Result:", res)

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
    asyncio.run(main())
