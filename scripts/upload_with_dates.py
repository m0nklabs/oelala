import os
from datetime import datetime, timezone
from pathlib import Path

import httpx


DEFAULT_SOURCES = [
    (Path("/home/flip/BACKUP-OELALA-MEDIA/media/generated"), "generated"),
    (Path("/home/flip/BACKUP-OELALA-MEDIA/ComfyUI/output"), "comfyui-local"),
]


def upload_with_dates() -> None:
    api_key = os.environ.get("OELALA_STORAGE_API_KEY")
    base_url = os.environ.get("OELALA_STORAGE_URL", "http://127.0.0.1:7990")

    if not api_key:
        raise RuntimeError("Set OELALA_STORAGE_API_KEY before running this script.")

    uploaded = 0

    for source_dir, bucket in DEFAULT_SOURCES:
        if not source_dir.exists():
            continue

        bucket_url = f"{base_url.rstrip('/')}/{bucket}"
        for file_path in source_dir.glob("*"):
            if not file_path.is_file():
                continue

            mtime = file_path.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            file_url = f"{bucket_url}/{file_path.name}"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-Modified-At": mtime_str,
            }

            with file_path.open("rb") as handle:
                data = handle.read()

            try:
                response = httpx.put(file_url, content=data, headers=headers, timeout=120.0)
                if response.status_code in (200, 201):
                    uploaded += 1
                    if uploaded % 50 == 0:
                        print(
                            f"Uploaded {uploaded} files... "
                            f"(last: {file_path.name} @ {mtime_str})"
                        )
                else:
                    print(f"Failed {file_path.name}: {response.status_code} {response.text}")
            except Exception as exc:
                print(f"Error {file_path.name}: {exc}")

    print(
        "DONE! Re-uploaded "
        f"{uploaded} files with their original creation/modification dates intact."
    )


if __name__ == "__main__":
    upload_with_dates()
