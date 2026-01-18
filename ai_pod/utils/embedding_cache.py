"""Embedding-specific caching utilities for SPECTER2 embeddings."""

import hashlib
import json
from pathlib import Path

from loguru import logger

from ai_pod.models import UserProfile


CACHE_DIR = Path(__file__).parent.parent.parent / "data"


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_paper_embeddings_path() -> Path:
    """Get the path to the paper embeddings cache file."""
    _ensure_cache_dir()
    return CACHE_DIR / "paper_embeddings.json"


def get_profile_hash(profile: UserProfile) -> str:
    """Generate a hash for a user profile based on its content."""
    content = {
        "topics": sorted(profile.topics),
        "keywords": sorted(profile.keywords),
        "past_papers": sorted(
            [{"title": p.title, "abstract": p.abstract or ""} for p in profile.past_papers],
            key=lambda x: x["title"],
        ),
    }
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()[:12]


def get_profile_embeddings_path(profile: UserProfile) -> Path:
    """Get the path to the profile embeddings cache file."""
    _ensure_cache_dir()
    profile_hash = get_profile_hash(profile)
    return CACHE_DIR / f"profile_embeddings_{profile_hash}.json"


def load_paper_embeddings() -> dict[str, list[float]] | None:
    """Load paper embeddings from cache.

    Returns:
        Dictionary mapping arxiv_id to embedding vector, or None if cache doesn't exist.
    """
    cache_path = get_paper_embeddings_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embeddings = data.get("embeddings", {})
        logger.debug(f"Loaded {len(embeddings)} paper embeddings from cache")
        return embeddings
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load paper embeddings cache: {e}")
        return None


def save_paper_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Save paper embeddings to cache.

    This merges new embeddings with existing cache (append-only).

    Args:
        embeddings: Dictionary mapping arxiv_id to embedding vector.
    """
    cache_path = get_paper_embeddings_path()

    # Load existing embeddings and merge
    existing = load_paper_embeddings() or {}
    existing.update(embeddings)

    data = {"embeddings": existing}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.debug(f"Saved {len(existing)} paper embeddings to cache")


def load_profile_embeddings(profile: UserProfile) -> dict[str, list[list[float]]] | None:
    """Load profile embeddings from cache.

    Args:
        profile: User profile to load embeddings for.

    Returns:
        Dictionary with 'topics', 'keywords', 'papers' keys mapping to lists of embeddings,
        or None if cache doesn't exist or is invalid.
    """
    cache_path = get_profile_embeddings_path(profile)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embeddings = data.get("embeddings", {})
        logger.debug(f"Loaded profile embeddings from cache for profile '{profile.name}'")
        return embeddings
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load profile embeddings cache: {e}")
        return None


def save_profile_embeddings(
    profile: UserProfile, embeddings: dict[str, list[list[float]]]
) -> None:
    """Save profile embeddings to cache.

    Args:
        profile: User profile the embeddings are for.
        embeddings: Dictionary with 'topics', 'keywords', 'papers' keys.
    """
    cache_path = get_profile_embeddings_path(profile)
    data = {
        "profile_name": profile.name,
        "profile_hash": get_profile_hash(profile),
        "embeddings": embeddings,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.debug(f"Saved profile embeddings to cache for profile '{profile.name}'")
