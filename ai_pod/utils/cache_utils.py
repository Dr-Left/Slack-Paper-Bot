"""Caching utilities for paper data."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from loguru import logger

from ai_pod.models import Author, Paper, ArxivCategory, OutputMode


CACHE_DIR = Path(__file__).parent.parent.parent / "data"


def get_cache_key(categories: list[ArxivCategory], days: int, mode: OutputMode) -> str:
    """Generate a unique cache key based on query parameters."""
    cat_str = "_".join(sorted(c.value for c in categories))
    key_str = f"{cat_str}_{days}_{mode.value}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def get_cache_path(cache_key: str) -> Path:
    """Get the cache file path for a given cache key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"papers_{cache_key}.json"


def paper_to_dict(paper: Paper) -> dict:
    """Convert Paper to JSON-serializable dict."""
    d = asdict(paper)
    d["published"] = paper.published.isoformat()
    d["updated"] = paper.updated.isoformat()
    return d


def dict_to_paper(d: dict) -> Paper:
    """Convert dict back to Paper object."""
    # Convert authors - handle both old format (list of strings) and new format (list of dicts)
    authors = []
    for author in d["authors"]:
        if isinstance(author, str):
            # Old format: just a name string
            authors.append(Author(name=author))
        else:
            # New format: dict with name and affiliation
            authors.append(Author(name=author["name"], affiliation=author.get("affiliation")))

    return Paper(
        arxiv_id=d["arxiv_id"],
        title=d["title"],
        abstract=d.get("abstract"),
        authors=authors,
        published=datetime.fromisoformat(d["published"]),
        updated=datetime.fromisoformat(d["updated"]),
        categories=d["categories"],
        pdf_url=d["pdf_url"],
    )


def load_cache(cache_key: str) -> tuple[list[Paper], datetime] | None:
    """Load papers from cache if available and valid."""
    cache_path = get_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached_at = datetime.fromisoformat(data["cached_at"])
        papers = [dict_to_paper(p) for p in data["papers"]]
        logger.debug(f"Loaded {len(papers)} papers from cache (cached at {cached_at})")
        return papers, cached_at
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def save_cache(cache_key: str, papers: list[Paper], categories: list[ArxivCategory], days: int, mode: OutputMode) -> None:
    """Save papers to cache."""
    cache_path = get_cache_path(cache_key)
    data = {
        "cached_at": datetime.now().isoformat(),
        "query": {
            "categories": [c.value for c in categories],
            "days": days,
            "mode": mode.value,
        },
        "papers": [paper_to_dict(p) for p in papers],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Saved {len(papers)} papers to cache at {cache_path}")

