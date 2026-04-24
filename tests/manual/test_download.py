import asyncio
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

from storage_client import get_storage_client

async def main():
    st = get_storage_client()
    # It might take a bit for the worker to start handling and processing. The webhook would hit the db, we bypass that.
    print("Checking MinIO...")
    files = st.list("oelala-users", prefix="97833cbb-ed5b-40f9-ab32-033877dcf77d/generations/dc9a5c61-edd7-4f5e-9def-2680ebd04b4a/")
    for f in files:
        print(f)

if __name__ == "__main__":
    asyncio.run(main())
