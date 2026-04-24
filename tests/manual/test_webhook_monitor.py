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
    res = st.list("oelala-generated", prefix="")
    for obj in sorted(res, key=lambda x: x.get('last_modified', ''), reverse=True)[:5]:
        print(obj)

if __name__ == "__main__":
    asyncio.run(main())
