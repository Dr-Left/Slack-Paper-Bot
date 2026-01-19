"""Posted papers tracking utilities."""

import json
from pathlib import Path

from loguru import logger

from ai_pod.models import FilteredPaper


POSTED_PAPERS_PATH = Path("data/posted_papers.json")


def load_posted_papers() -> list[dict]:
    """Load list of previously posted papers with metadata.

    Returns:
        List of paper dicts with arxiv_id, message_ts, title, abstract.
        Handles backward compatibility with old format (just IDs).
    """
    if not POSTED_PAPERS_PATH.exists():
        return []

    with open(POSTED_PAPERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle new format
    if "posted_papers" in data:
        return data["posted_papers"]

    # Backward compatibility: old format had "posted_ids"
    # Convert to new format (without message_ts or metadata)
    old_ids = data.get("posted_ids", [])
    return [{"arxiv_id": arxiv_id} for arxiv_id in old_ids]


def get_posted_paper_ids() -> set[str]:
    """Get set of previously posted paper IDs.

    Returns:
        Set of arxiv_ids that have been posted.
    """
    papers = load_posted_papers()
    return {p["arxiv_id"] for p in papers}


def save_posted_papers(new_papers: list[dict]) -> None:
    """Save newly posted papers with metadata to the tracking file.

    Args:
        new_papers: List of paper dicts with arxiv_id, message_ts, title, abstract.
    """
    POSTED_PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_posted_papers()
    existing_ids = {p["arxiv_id"] for p in existing}

    # Add new papers (avoid duplicates)
    for paper in new_papers:
        if paper["arxiv_id"] not in existing_ids:
            existing.append(paper)
            existing_ids.add(paper["arxiv_id"])

    with open(POSTED_PAPERS_PATH, "w", encoding="utf-8") as f:
        json.dump({"posted_papers": existing}, f, indent=2)

    logger.info(f"Saved {len(new_papers)} new papers to posted papers tracking")


def filter_already_posted(papers: list[FilteredPaper]) -> list[FilteredPaper]:
    """Remove papers that have already been posted.

    Args:
        papers: List of filtered papers.

    Returns:
        Papers that haven't been posted yet.
    """
    posted_ids = get_posted_paper_ids()
    filtered = [p for p in papers if p.paper.arxiv_id not in posted_ids]

    if len(filtered) < len(papers):
        logger.info(f"Filtered out {len(papers) - len(filtered)} already-posted papers")

    return filtered
