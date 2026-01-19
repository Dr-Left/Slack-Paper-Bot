"""Slack formatting, posting, and reaction handling utilities."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from loguru import logger

from ai_pod.filter_papers import load_user_profile, save_user_profile
from ai_pod.models import FilteredPaper, PastPaper
from ai_pod.posted_papers import load_posted_papers, save_posted_papers

if TYPE_CHECKING:
    from ai_pod.slack_bot import SlackConfig


def format_paper_message(paper: FilteredPaper, index: int) -> str:
    """Format a single paper for Slack display.

    Args:
        paper: FilteredPaper object.
        index: Paper number in the list.

    Returns:
        Formatted string for Slack.
    """
    p = paper.paper
    arxiv_url = f"https://arxiv.org/abs/{p.arxiv_id}"

    # Get first 2-3 authors
    author_names = [a.name for a in p.authors[:3]]
    authors_str = ", ".join(author_names)
    if len(p.authors) > 3:
        authors_str += " et al."

    return f"*{index}. [{paper.similarity_score:.2f}]* <{arxiv_url}|{p.title}>\n   by {authors_str}"


def format_header_blocks(num_papers: int) -> list[dict]:
    """Format the daily digest header as Slack blocks.

    Args:
        num_papers: Number of papers being posted.

    Returns:
        List of Slack block elements for the header.
    """
    today = datetime.now().strftime("%B %d, %Y")

    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Top {num_papers} Papers for {today}",
                "emoji": True,
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "React with :fire: to add a paper to your profile for better recommendations!",
                }
            ],
        },
        {"type": "divider"},
    ]


def format_single_paper_blocks(paper: FilteredPaper, index: int) -> list[dict]:
    """Format a single paper as Slack blocks.

    Args:
        paper: FilteredPaper object.
        index: Paper number in the list.

    Returns:
        List of Slack block elements for the paper.
    """
    p = paper.paper
    arxiv_url = f"https://arxiv.org/abs/{p.arxiv_id}"

    # Get first 2-3 authors
    author_names = [a.name for a in p.authors[:3]]
    authors_str = ", ".join(author_names)
    if len(p.authors) > 3:
        authors_str += " et al."

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{index}. [{paper.similarity_score:.2f}]* <{arxiv_url}|{p.title}>\nby {authors_str}",
            },
        },
    ]

    return blocks


def format_summary_footer_blocks(summary: str) -> list[dict]:
    """Format the summary as Slack blocks for the footer.

    Args:
        summary: The generated summary text.

    Returns:
        List of Slack block elements for the footer.
    """
    def _truncate_to_char_limit_preserving_lines(text: str, limit: int) -> str:
        """Best-effort truncate without cutting mid-line (avoids breaking Slack link markup)."""
        if limit <= 0:
            return ""
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        lines = text.splitlines()
        out_lines: list[str] = []
        remaining = limit - 4  # space for "\n..."
        for line in lines:
            if not line:
                candidate = ""
            else:
                candidate = line
            # +1 accounts for newline join when there are existing lines
            extra = len(candidate) + (1 if out_lines else 0)
            if extra > remaining:
                break
            out_lines.append(candidate)
            remaining -= extra
        truncated = "\n".join(out_lines).rstrip()
        if not truncated:
            return (text[: max(0, limit - 3)] + "...").rstrip()
        return (truncated + "\n...").rstrip()

    # Slack section.text has a hard limit; keep this safely below it.
    body_text = _truncate_to_char_limit_preserving_lines(
        f"*Today's Digest Summary*\n{(summary or '').strip()}",
        limit=2900,
    )
    return [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": body_text,
            },
        },
    ]


def post_to_slack(
    config: SlackConfig, papers: list[FilteredPaper]
) -> list[tuple[str, str]]:
    """Post papers to Slack channel, each as a separate message.

    Args:
        config: Slack configuration.
        papers: List of papers to post.

    Returns:
        List of (arxiv_id, message_ts) tuples for successfully posted papers.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=config.bot_token)
    posted: list[tuple[str, str]] = []

    # Post header message first
    try:
        header_blocks = format_header_blocks(len(papers))
        client.chat_postMessage(
            channel=config.channel_id,
            blocks=header_blocks,
            text=f"Top {len(papers)} ML papers for today",
        )
    except SlackApiError as e:
        logger.error(f"Failed to post header: {e.response['error']}")
        return posted

    # Post each paper as a separate message
    for i, paper in enumerate(papers, 1):
        try:
            blocks = format_single_paper_blocks(paper, i)
            response = client.chat_postMessage(
                channel=config.channel_id,
                blocks=blocks,
                text=f"{i}. {paper.paper.title}",  # Fallback for notifications
            )
            message_ts = response["ts"]
            posted.append((paper.paper.arxiv_id, message_ts))
            logger.debug(f"Posted paper {paper.paper.arxiv_id}: {message_ts}")

        except SlackApiError as e:
            logger.error(f"Failed to post paper {paper.paper.arxiv_id}: {e.response['error']}")
            continue

    logger.info(f"Posted {len(posted)}/{len(papers)} papers to Slack channel {config.channel_id}")
    return posted


def post_summary_footer(config: SlackConfig, summary: str) -> bool:
    """Post the summary footer to Slack.

    Args:
        config: Slack configuration.
        summary: The generated summary text.

    Returns:
        True if successfully posted, False otherwise.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=config.bot_token)

    try:
        blocks = format_summary_footer_blocks(summary)
        client.chat_postMessage(
            channel=config.channel_id,
            blocks=blocks,
            text="Today's Digest Summary",
        )
        logger.info("Posted summary footer to Slack")
        return True

    except SlackApiError as e:
        logger.error(f"Failed to post summary footer: {e.response['error']}")
        return False


def fetch_reactions_and_update_profile(
    config: SlackConfig,
    reaction_emoji: str = "fire",
) -> int:
    """Fetch reactions from posted papers and add liked papers to profile.

    Args:
        config: Slack configuration.
        reaction_emoji: The emoji reaction to look for (default: "fire" for fire emoji).

    Returns:
        Number of papers added to the profile.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    posted_papers = load_posted_papers()

    # Filter to papers that have message_ts (can check reactions)
    papers_with_ts = [p for p in posted_papers if p.get("message_ts")]

    if not papers_with_ts:
        logger.debug("No posted papers with message_ts to check for reactions")
        return 0

    logger.info(f"Checking reactions on {len(papers_with_ts)} posted papers...")

    client = WebClient(token=config.bot_token)

    # Load profile to check existing papers and add new ones
    profile = load_user_profile(config.profile_path)
    existing_titles = {p.title.lower() for p in profile.past_papers}
    existing_arxiv_ids = {p.arxiv_id for p in profile.past_papers if p.arxiv_id}

    papers_to_add = []

    for paper in papers_with_ts:
        arxiv_id = paper.get("arxiv_id")
        message_ts = paper.get("message_ts")
        title = paper.get("title")
        abstract = paper.get("abstract")

        # Skip if already in profile (by title or arxiv_id)
        if title and title.lower() in existing_titles:
            continue
        if arxiv_id and arxiv_id in existing_arxiv_ids:
            continue

        # Check for reactions on this message
        try:
            response = client.reactions_get(
                channel=config.channel_id,
                timestamp=message_ts,
            )

            # Check if the target reaction exists
            message = response.get("message", {})
            reactions = message.get("reactions", [])

            has_target_reaction = any(r.get("name") == reaction_emoji for r in reactions)

            if has_target_reaction and title:
                papers_to_add.append(
                    PastPaper(title=title, abstract=abstract, arxiv_id=arxiv_id)
                )
                logger.debug(f"Found fire reaction on: {title}")

        except SlackApiError as e:
            logger.warning(f"Failed to get reactions for {arxiv_id}: {e.response['error']}")
            continue

    # Add papers to profile
    if papers_to_add:
        profile.past_papers.extend(papers_to_add)
        save_user_profile(profile, config.profile_path)
        logger.info(f"Added {len(papers_to_add)} papers to profile from reactions")

    return len(papers_to_add)


def extract_arxiv_ids_from_text(text: str) -> list[str]:
    """Extract arXiv IDs from text containing arxiv.org URLs.

    Args:
        text: Text to search for arXiv URLs.

    Returns:
        List of unique arXiv IDs found.
    """
    # Match arxiv.org/abs/<id> or arxiv.org/pdf/<id>
    pattern = r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Remove version suffixes (v1, v2, etc.) for consistency
    ids = [re.sub(r'v[0-9]+$', '', match) for match in matches]

    return list(set(ids))  # Return unique IDs


def fetch_paper_metadata_from_arxiv_txt(arxiv_id: str) -> Optional[dict]:
    """Fetch paper metadata from alternative sources as fallback.

    Tries multiple fallback sources:
    1. arxiv-txt.org - plain text version
    2. arXiv HTML abstract page - scraping

    Args:
        arxiv_id: arXiv ID to fetch.

    Returns:
        Dict with title and abstract, or None if fetch fails.
    """
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError

    # Try 1: arxiv-txt.org
    try:
        url = f"https://www.arxiv-txt.org/txt/{arxiv_id}"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urlopen(req, timeout=15) as response:
            text_data = response.read().decode('utf-8')

        # Parse the plain text
        lines = text_data.strip().split('\n')
        if lines:
            title = None
            abstract = None

            # Look for title (first substantial line)
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    title = line
                    break

            # Look for abstract section
            for i, line in enumerate(lines):
                if re.search(r'\babstract\b', line, re.IGNORECASE):
                    abstract_lines = []
                    for j in range(i + 1, min(i + 20, len(lines))):
                        line = lines[j].strip()
                        if line:
                            if re.match(r'^(Introduction|1\.|I\.)', line, re.IGNORECASE):
                                break
                            abstract_lines.append(line)
                    if abstract_lines:
                        abstract = " ".join(abstract_lines)
                    break

            if title:
                logger.info(f"Fetched metadata from arxiv-txt.org for {arxiv_id}")
                return {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract or "",
                }

    except (HTTPError, URLError) as e:
        logger.debug(f"arxiv-txt.org failed for {arxiv_id}: {e}")

    # Try 2: Scrape arXiv HTML abstract page
    try:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urlopen(req, timeout=15) as response:
            html_data = response.read().decode('utf-8')

        # Extract title from HTML
        title_match = re.search(r'<meta name="citation_title" content="([^"]+)"', html_data)
        title = title_match.group(1) if title_match else None

        # Extract abstract from HTML
        abstract_match = re.search(
            r'<blockquote class="abstract[^"]*"[^>]*>.*?<span class="descriptor">Abstract:</span>\s*(.+?)</blockquote>',
            html_data,
            re.DOTALL
        )
        abstract = None
        if abstract_match:
            abstract = re.sub(r'<[^>]+>', '', abstract_match.group(1))  # Remove HTML tags
            abstract = " ".join(abstract.strip().split())  # Clean whitespace

        if title:
            logger.info(f"Fetched metadata from arXiv HTML for {arxiv_id}")
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract or "",
            }

    except (HTTPError, URLError) as e:
        logger.debug(f"arXiv HTML scraping failed for {arxiv_id}: {e}")

    logger.debug(f"All fallback methods failed for {arxiv_id}")
    return None


def fetch_paper_metadata_from_arxiv(arxiv_id: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch paper metadata from arXiv API with retry logic and fallback.

    Args:
        arxiv_id: arXiv ID to fetch.
        max_retries: Maximum number of retry attempts for rate limiting.

    Returns:
        Dict with title and abstract, or None if fetch fails.
    """
    import xml.etree.ElementTree as ET
    from urllib.request import urlopen
    from urllib.error import URLError, HTTPError

    # Try official arXiv API first
    for attempt in range(max_retries):
        try:
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            with urlopen(url, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)

            # Parse Atom feed
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)

            if entry is None:
                logger.warning(f"No entry found for arXiv ID {arxiv_id}")
                # Try fallback
                return fetch_paper_metadata_from_arxiv_txt(arxiv_id)

            title = entry.find('atom:title', ns)
            abstract = entry.find('atom:summary', ns)

            if title is not None and abstract is not None:
                # Clean up whitespace
                title_text = " ".join(title.text.strip().split())
                abstract_text = " ".join(abstract.text.strip().split())

                return {
                    "arxiv_id": arxiv_id,
                    "title": title_text,
                    "abstract": abstract_text,
                }

            # Try fallback if data incomplete
            return fetch_paper_metadata_from_arxiv_txt(arxiv_id)

        except HTTPError as e:
            if e.code == 429:  # Rate limit error
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                logger.warning(f"Rate limited for {arxiv_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"HTTP error {e.code} for {arxiv_id}, trying fallback")
                # Try fallback on HTTP errors
                return fetch_paper_metadata_from_arxiv_txt(arxiv_id)

        except (URLError, ET.ParseError) as e:
            logger.warning(f"Failed to fetch metadata for {arxiv_id}: {e}, trying fallback")
            # Try fallback on other errors
            return fetch_paper_metadata_from_arxiv_txt(arxiv_id)

    logger.warning(f"Failed to fetch metadata for {arxiv_id} after {max_retries} retries, trying fallback")
    # Try fallback after exhausting retries
    return fetch_paper_metadata_from_arxiv_txt(arxiv_id)


def import_papers_from_channel(
    config: SlackConfig,
    days: Optional[int] = None,
    fetch_metadata: bool = True,
    channel_id: Optional[str] = None,
) -> int:
    """Import papers from Slack channel history.

    Scans channel messages for arXiv URLs and adds them to posted papers tracking.

    Args:
        config: Slack configuration.
        days: Number of days of history to scan (None = all history).
        fetch_metadata: If True, fetch paper metadata from arXiv API.
        channel_id: Optional channel ID to import from (overrides config.channel_id).

    Returns:
        Number of papers imported.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=config.bot_token)

    # Use provided channel_id or fall back to config
    target_channel = channel_id if channel_id else config.channel_id

    # Calculate oldest timestamp if days is specified
    oldest = None
    if days is not None:
        oldest_dt = datetime.now() - timedelta(days=days)
        oldest = str(oldest_dt.timestamp())

    logger.info(f"Scanning Slack channel {target_channel} for arXiv papers...")
    if days:
        logger.info(f"Looking back {days} days")
    else:
        logger.info("Scanning all history")

    all_arxiv_ids = set()
    paper_data = {}  # arxiv_id -> {message_ts, title, abstract}

    try:
        # Fetch channel history with pagination
        cursor = None
        message_count = 0

        while True:
            response = client.conversations_history(
                channel=target_channel,
                oldest=oldest,
                cursor=cursor,
                limit=200,  # Max per request
            )

            messages = response.get("messages", [])
            message_count += len(messages)

            # Extract arXiv IDs from each message
            for msg in messages:
                text = msg.get("text", "")

                # Also check blocks for links
                blocks = msg.get("blocks", [])
                for block in blocks:
                    if block.get("type") == "section":
                        block_text = block.get("text", {}).get("text", "")
                        text += " " + block_text

                arxiv_ids = extract_arxiv_ids_from_text(text)

                for arxiv_id in arxiv_ids:
                    all_arxiv_ids.add(arxiv_id)
                    # Store message timestamp for the first occurrence
                    if arxiv_id not in paper_data:
                        paper_data[arxiv_id] = {
                            "arxiv_id": arxiv_id,
                            "message_ts": msg.get("ts"),
                        }

            # Check if there are more messages
            if not response.get("has_more"):
                break

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

            # Rate limiting
            time.sleep(1)

        logger.info(f"Scanned {message_count} messages, found {len(all_arxiv_ids)} unique arXiv papers")

    except SlackApiError as e:
        logger.error(f"Failed to fetch channel history: {e.response['error']}")
        return 0

    if not all_arxiv_ids:
        logger.info("No arXiv papers found in channel")
        return 0

    # Fetch metadata from arXiv if requested
    if fetch_metadata:
        logger.info(f"Fetching paper metadata from arXiv (this will take ~{len(all_arxiv_ids) * 5 / 60:.1f} minutes)...")
        for i, arxiv_id in enumerate(all_arxiv_ids, 1):
            logger.info(f"Fetching metadata {i}/{len(all_arxiv_ids)}: {arxiv_id}")
            metadata = fetch_paper_metadata_from_arxiv(arxiv_id)
            if metadata:
                paper_data[arxiv_id].update({
                    "title": metadata["title"],
                    "abstract": metadata["abstract"],
                })
                logger.debug(f"Fetched metadata for {arxiv_id}: {metadata['title']}")
            else:
                logger.warning(f"Could not fetch metadata for {arxiv_id}, saving with ID only")

            # Rate limiting for arXiv API (recommended: 1 request per 3-5 seconds)
            # Using 5 seconds to be safe and avoid 429 errors
            if i < len(all_arxiv_ids):
                time.sleep(5)

    # Save to posted papers tracking
    papers_to_save = list(paper_data.values())
    save_posted_papers(papers_to_save)

    logger.info(f"Successfully imported {len(papers_to_save)} papers from channel")
    return len(papers_to_save)
