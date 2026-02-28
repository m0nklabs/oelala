"""
Face swap and face profile service for oelala.

Uses deepinsight/insightface directly (no ComfyUI needed for swap).
Supports:
  - Face swap via inswapper_128.onnx
  - Face profile storage (multi-image reference per profile)
  - Embedding extraction for fast profile lookup

Model paths:
  - Analyzer (buffalo_l): ComfyUI/models/insightface/models/buffalo_l/
  - Swapper:              ComfyUI/models/insightface/inswapper_128.onnx
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEBUG = os.getenv("DEBUG", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Base dir is two levels up from this file → /home/flip/oelala/
BACKEND_DIR = Path(__file__).parent
BASE_DIR = BACKEND_DIR.parent.parent

INSIGHTFACE_MODEL_ROOT = BASE_DIR / "ComfyUI" / "models" / "insightface"
INSWAPPER_PATH = INSIGHTFACE_MODEL_ROOT / "inswapper_128.onnx"

FACE_PROFILES_DIR = BASE_DIR / "data" / "face_profiles"
FACE_PROFILES_INDEX = FACE_PROFILES_DIR / "index.json"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton models
# ─────────────────────────────────────────────────────────────────────────────


class _NullWriter:
    """Suppress insightface stdout spam."""

    def write(self, _):
        pass

    def flush(self):
        pass


_face_analyser = None
_face_swapper = None


def _get_analyser():
    """Lazy-load insightface FaceAnalysis (buffalo_l)."""
    global _face_analyser
    if _face_analyser is not None:
        return _face_analyser

    try:
        import insightface

        INSIGHTFACE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"🐛 Loading insightface buffalo_l from {INSIGHTFACE_MODEL_ROOT}"
        )
        _face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=INSIGHTFACE_MODEL_ROOT.as_posix(),
        )
        _face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✅ InsightFace analyser loaded")
        return _face_analyser
    except Exception as e:
        logger.error(f"❌ Failed to load face analyser: {e}")
        raise RuntimeError(f"InsightFace load failed: {e}") from e


def _get_swapper():
    """Lazy-load inswapper_128.onnx model."""
    global _face_swapper
    if _face_swapper is not None:
        return _face_swapper

    if not INSWAPPER_PATH.exists():
        raise FileNotFoundError(
            f"inswapper_128.onnx not found at {INSWAPPER_PATH}. "
            "Download it from https://huggingface.co/ezioruan/inswapper_128.onnx"
        )

    try:
        import insightface

        logger.info(f"🐛 Loading face swapper from {INSWAPPER_PATH}")
        _face_swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_PATH),
            download=False,
        )
        _face_swapper.prepare(ctx_id=0)
        logger.info("✅ Face swapper (inswapper_128) loaded")
        return _face_swapper
    except Exception as e:
        logger.error(f"❌ Failed to load face swapper: {e}")
        raise RuntimeError(f"Swapper load failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Core face detection
# ─────────────────────────────────────────────────────────────────────────────


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array for insightface."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array back to PIL Image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _get_face_single(
    analyser,
    cv_img: np.ndarray,
    face_index: int = 0,
    det_size: tuple[int, int] = (640, 640),
):
    """Get a single face from image, retry with smaller det_size if needed."""
    analyser.prepare(ctx_id=0, det_size=det_size)
    faces = analyser.get(cv_img)

    if len(faces) == 0 and det_size[0] > 320:
        return _get_face_single(
            analyser, cv_img, face_index, (det_size[0] // 2, det_size[1] // 2)
        )

    try:
        return sorted(faces, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def detect_faces(img: Image.Image | bytes) -> list[dict]:
    """
    Detect all faces in an image.

    Returns list of dicts with: index, bbox (x,y,w,h), confidence.
    """
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))

    analyser = _get_analyser()
    cv_img = _pil_to_bgr(img)
    analyser.prepare(ctx_id=0, det_size=(640, 640))
    faces = analyser.get(cv_img)

    result = []
    for i, face in enumerate(sorted(faces, key=lambda x: x.bbox[0])):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        result.append(
            {
                "index": i,
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "confidence": float(face.det_score),
            }
        )

    logger.info(f"👤 Detected {len(result)} face(s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Face Swap
# ─────────────────────────────────────────────────────────────────────────────


def swap_faces(
    source_img: Image.Image | bytes,
    target_img: Image.Image | bytes,
    face_indices: list[int] | None = None,
) -> Image.Image:
    """
    Swap face(s) from source_img into target_img.

    Args:
        source_img: Reference image with the source face.
        target_img: Target image where face(s) will be replaced.
        face_indices: Which face indices in target to replace. None = [0].

    Returns:
        PIL Image with swapped faces.
    """
    if isinstance(source_img, bytes):
        source_img = Image.open(io.BytesIO(source_img))
    if isinstance(target_img, bytes):
        target_img = Image.open(io.BytesIO(target_img))

    if face_indices is None:
        face_indices = [0]

    analyser = _get_analyser()
    swapper = _get_swapper()

    cv_source = _pil_to_bgr(source_img)
    cv_target = _pil_to_bgr(target_img)

    # Get source face embedding
    source_face = _get_face_single(analyser, cv_source, face_index=0)
    if source_face is None:
        raise ValueError("No face detected in source image")

    result = cv_target.copy()

    for face_idx in face_indices:
        target_face = _get_face_single(analyser, cv_target, face_index=face_idx)
        if target_face is None:
            logger.warning(f"⚠️ No face found at index {face_idx} in target")
            continue

        # Silence insightface stdout
        sys.stdout = _NullWriter()
        try:
            result = swapper.get(result, target_face, source_face, paste_back=True)
        finally:
            sys.stdout = sys.__stdout__

        logger.info(f"✅ Swapped face {face_idx}")

    return _bgr_to_pil(result)


def swap_faces_to_bytes(
    source_img: Image.Image | bytes,
    target_img: Image.Image | bytes,
    face_indices: list[int] | None = None,
    output_format: str = "PNG",
) -> bytes:
    """Swap faces and return result as image bytes."""
    result = swap_faces(source_img, target_img, face_indices)
    buf = io.BytesIO()
    result.save(buf, format=output_format)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Face Profiles (multi-image reference profiles per person)
# ─────────────────────────────────────────────────────────────────────────────


def _load_profiles_index() -> dict:
    """Load profiles index from disk."""
    if not FACE_PROFILES_INDEX.exists():
        return {}
    with open(FACE_PROFILES_INDEX) as f:
        return json.load(f)


def _save_profiles_index(profiles: dict) -> None:
    """Save profiles index to disk."""
    FACE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FACE_PROFILES_INDEX, "w") as f:
        json.dump(profiles, f, indent=2)


def list_face_profiles() -> list[dict]:
    """Return all face profiles (without raw image data)."""
    profiles = _load_profiles_index()
    return list(profiles.values())


def get_face_profile(profile_id: str) -> dict | None:
    """Get a single face profile by ID."""
    profiles = _load_profiles_index()
    return profiles.get(profile_id)


def create_face_profile(
    name: str,
    images: list[bytes],
    description: str = "",
) -> dict:
    """
    Create a new face profile from one or more reference images.

    Extracts and averages face embeddings across all input images for
    better identity stability.

    Returns the profile metadata dict.
    """
    if not images:
        raise ValueError("At least one reference image required")

    profile_id = uuid.uuid4().hex[:12]
    profile_dir = FACE_PROFILES_DIR / profile_id
    images_dir = profile_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    analyser = _get_analyser()

    embeddings = []
    saved_images = []

    for i, img_bytes in enumerate(images):
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            cv_img = _pil_to_bgr(img)
            analyser.prepare(ctx_id=0, det_size=(640, 640))
            faces = analyser.get(cv_img)

            if not faces:
                logger.warning(f"⚠️ No face in image {i}, skipping")
                continue

            # Pick highest-confidence face
            face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
            embeddings.append(face.normed_embedding)

            # Save image
            fname = f"ref_{i:03d}.jpg"
            img_path = images_dir / fname
            img.save(img_path, format="JPEG", quality=85)
            saved_images.append(fname)

        except Exception as e:
            logger.warning(f"⚠️ Failed to process image {i}: {e}")

    if not embeddings:
        raise ValueError(
            "No valid faces detected in any of the provided images"
        )

    # Average embeddings across all images for stable identity
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # normalize

    # Save embedding
    embedding_path = profile_dir / "embedding.npy"
    np.save(str(embedding_path), avg_embedding)

    profile = {
        "id": profile_id,
        "name": name,
        "description": description,
        "images": saved_images,
        "image_count": len(saved_images),
        "embedding_count": len(embeddings),
    }

    # Update index
    profiles = _load_profiles_index()
    profiles[profile_id] = profile
    _save_profiles_index(profiles)

    logger.info(
        f"✅ Created face profile '{name}' (id={profile_id}) "
        f"from {len(embeddings)} images"
    )
    return profile


def delete_face_profile(profile_id: str) -> bool:
    """
    Delete a face profile and all its associated files.

    Returns True if deleted, False if not found.
    """
    import shutil

    profiles = _load_profiles_index()
    if profile_id not in profiles:
        return False

    # Delete files
    profile_dir = FACE_PROFILES_DIR / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)

    # Update index
    del profiles[profile_id]
    _save_profiles_index(profiles)

    logger.info(f"🗑️ Deleted face profile {profile_id}")
    return True


def swap_with_profile(
    target_img: Image.Image | bytes,
    profile_id: str,
    face_indices: list[int] | None = None,
) -> Image.Image:
    """
    Swap faces in target_img using a saved face profile as source.

    Uses the first reference image from the profile.

    Args:
        target_img: Target image where face(s) will be replaced.
        profile_id: ID of the saved face profile to use as source.
        face_indices: Which faces in target to swap.

    Returns:
        PIL Image with swapped faces.
    """
    profile = get_face_profile(profile_id)
    if not profile:
        raise ValueError(f"Face profile '{profile_id}' not found")

    if not profile["images"]:
        raise ValueError(f"Face profile '{profile_id}' has no reference images")

    # Load first reference image as source
    source_path = (
        FACE_PROFILES_DIR / profile_id / "images" / profile["images"][0]
    )
    with open(source_path, "rb") as f:
        source_bytes = f.read()

    return swap_faces(source_bytes, target_img, face_indices)
