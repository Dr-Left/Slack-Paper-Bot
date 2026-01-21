"""Fetch recent papers from arXiv API."""

import argparse
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import time
import xml.etree.ElementTree as ET

from loguru import logger

from ai_pod.models import ArxivCategory, Author, OutputMode, Paper
from ai_pod.utils.cache_utils import (
    get_cache_key,
    load_cache,
    save_cache,
)


ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_PAGE_SIZE = 256  # don't set too large.
# atom is for xml parsing


def _parse_arxiv_response(xml_content: str, mode: OutputMode) -> list[Paper]:
    """Parse arXiv API XML response into Paper objects."""
    root = ET.fromstring(xml_content)
    papers = []

    for entry in root.findall("atom:entry", ATOM_NS):
        # Extract arxiv ID from the id URL
        id_elem = entry.find("atom:id", ATOM_NS)
        if id_elem is None or id_elem.text is None:
            continue
        arxiv_id = id_elem.text.split("/abs/")[-1]

        # Title
        title_elem = entry.find("atom:title", ATOM_NS)
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

        # Abstract (only if mode requires it)
        abstract = None
        if mode == OutputMode.TITLE_ABSTRACT:
            summary_elem = entry.find("atom:summary", ATOM_NS)
            if summary_elem is not None and summary_elem.text:
                abstract = summary_elem.text.strip().replace("\n", " ")

        # Authors with affiliations
        authors = []
        for author in entry.findall("atom:author", ATOM_NS):
            name_elem = author.find("atom:name", ATOM_NS)
            if name_elem is not None and name_elem.text:
                # Extract affiliation if available
                affiliation_elem = author.find("arxiv:affiliation", ATOM_NS)
                affiliation = None
                if affiliation_elem is not None and affiliation_elem.text:
                    affiliation = affiliation_elem.text.strip()
                authors.append(Author(name=name_elem.text, affiliation=affiliation))

        # Dates
        published_elem = entry.find("atom:published", ATOM_NS)
        updated_elem = entry.find("atom:updated", ATOM_NS)
        published = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00")) if published_elem is not None and published_elem.text else datetime.now()
        updated = datetime.fromisoformat(updated_elem.text.replace("Z", "+00:00")) if updated_elem is not None and updated_elem.text else datetime.now()

        # Categories
        categories = []
        for category in entry.findall("atom:category", ATOM_NS):
            term = category.get("term")
            if term:
                categories.append(term)

        # PDF URL
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=published,
            updated=updated,
            categories=categories,
            pdf_url=pdf_url,
        ))

    return papers


def get_papers(
    categories: list[ArxivCategory],
    days: int = 7,
    max_results: int = 100,
    mode: OutputMode = OutputMode.TITLE_ABSTRACT,
    use_cache: bool = True,
    cache_ttl_hours: float = 6.0,
) -> list[Paper]:
    """
    Fetch recent papers from arXiv.

    Args:
        categories: List of arXiv categories to search.
        days: Number of days back to search (papers published within this period).
        max_results: Maximum number of results to return; -1 for no limit.
        mode: Output mode - TITLE_ONLY or TITLE_ABSTRACT.
        use_cache: Whether to use caching (default: True).
        cache_ttl_hours: Cache time-to-live in hours (default: 6.0).

    Returns:
        List of Paper objects matching the criteria.
    """
    cache_key = get_cache_key(categories, days, mode)

    # Try to load from cache
    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            papers, cached_at = cached
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            if age_hours < cache_ttl_hours:
                logger.info(f"Using cached data ({age_hours:.1f}h old, TTL: {cache_ttl_hours}h)")
                if max_results == -1:
                    return papers
                return papers[:max_results]
            else:
                logger.info(f"Cache expired ({age_hours:.1f}h old, TTL: {cache_ttl_hours}h)")

    # Build category query (OR between categories)
    cat_query = " OR ".join(f"cat:{cat.value}" for cat in categories)
    query = f"({cat_query})"

    # Calculate date range for filtering
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    page_size = ARXIV_PAGE_SIZE if max_results == -1 else max_results
    all_papers: list[Paper] = []
    start = 0

    while True:
        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        logger.debug(f"Sending request to {url}")
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_content = response.read().decode("utf-8")
            logger.debug(f"Received response from {url}")
            logger.debug(xml_content)

        batch = _parse_arxiv_response(xml_content, mode)
        if not batch:
            break
        all_papers.extend(batch)

        if max_results == -1:
            if len(batch) < ARXIV_PAGE_SIZE:
                break
            start += ARXIV_PAGE_SIZE
        else:
            break
        if all_papers[-1].published.replace(tzinfo=None) < start_date:
            break
        time.sleep(1)

    filtered_papers = [
        p for p in all_papers
        if p.published.replace(tzinfo=None) >= start_date
    ]

    # Save to cache
    if use_cache and filtered_papers:
        save_cache(cache_key, filtered_papers, categories, days, mode)

    if max_results == -1:
        return filtered_papers
    return filtered_papers[:max_results]


def main():
    """CLI entry point for testing."""
    parser = argparse.ArgumentParser(description="Fetch recent papers from arXiv")

    # Category choices
    category_choices = [cat.value for cat in ArxivCategory]
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        choices=category_choices,
        default=["cs.LG", "cs.AI", "cs.CL"],
        help="arXiv categories to search (default: cs.LG cs.AI cs.CL)",
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=7,
        help="Number of days back to search (default: 7)",
    )
    parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=20,
        help="Maximum number of results (default: 20); -1 for no limit",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["title_only", "title_abstract"],
        default="title_abstract",
        help="Output mode (default: title_abstract)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (fetch fresh data)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=6.0,
        help="Cache TTL in hours (default: 6.0)",
    )
    parser.add_argument(
        "--show-affiliations",
        action="store_true",
        help="Show author affiliations when available",
    )

    args = parser.parse_args()

    if args.list_categories:
        print("Available arXiv categories:")
        for cat in ArxivCategory:
            print(f"  {cat.value}: {cat.name}")
        return

    # Convert string categories to enum
    categories = [ArxivCategory(cat) for cat in args.categories]
    mode = OutputMode(args.mode)

    print(f"Fetching papers from {args.categories} (last {args.days} days)...")
    print("-" * 60)

    papers = get_papers(
        categories=categories,
        days=args.days,
        max_results=args.max_results,
        mode=mode,
        use_cache=not args.no_cache,
        cache_ttl_hours=args.cache_ttl,
    )

    if not papers:
        print("No papers found matching criteria.")
        return

    print(f"Found {len(papers)} papers:\n")

    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper}")
        if args.show_affiliations:
            # Show authors with affiliations
            authors_with_affil = [str(a) for a in paper.authors if a.affiliation]
            if authors_with_affil:
                print(f"  Affiliations: {', '.join(authors_with_affil)}")
        if mode == OutputMode.TITLE_ABSTRACT and paper.abstract:
            # Truncate abstract for display
            abstract = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
            print(f"  Abstract: {abstract}")
        print()


if __name__ == "__main__":
    main()
