"""Slack bot for posting daily paper digests."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from typing import Optional

from ai_pod.filter_papers import filter_papers, load_user_profile, save_user_profile
from ai_pod.get_papers import get_papers
from ai_pod.models import ArxivCategory, FilteredPaper, OutputMode, PastPaper

from scholar_inbox.scholar_fetcher import fetch_scholar_inbox_data

@dataclass
class SlackConfig:
    """Configuration for the Slack bot."""

    bot_token: str
    channel_id: str
    profile_path: str
    categories: list[str]
    days: int
    top_k: int


POSTED_PAPERS_PATH = Path("data/posted_papers.json")


def load_config(config_path: str = "config/config.json") -> SlackConfig:
    """Load bot configuration from JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        SlackConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy config/config.example.json to config/config.json and fill in your values."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["bot_token", "channel_id"]
    for field in required_fields:
        if field not in data or not data[field]:
            raise ValueError(f"Missing required config field: {field}")

    return SlackConfig(
        bot_token=data["bot_token"],
        channel_id=data["channel_id"],
        profile_path=data.get("profile_path", "profiles/efficient_ml.json"),
        categories=data.get("categories", ["cs.LG", "cs.AI", "cs.CL"]),
        days=data.get("days", 1),
        top_k=data.get("top_k", 5),
    )


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


def fetch_reactions_and_update_profile(
    config: SlackConfig,
    reaction_emoji: str = "fire",
) -> int:
    """Fetch reactions from posted papers and add liked papers to profile.

    Args:
        config: Slack configuration.
        reaction_emoji: The emoji reaction to look for (default: "fire" for 🔥).

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
                logger.debug(f"Found 🔥 reaction on: {title}")

        except SlackApiError as e:
            logger.warning(f"Failed to get reactions for {arxiv_id}: {e.response['error']}")
            continue

    # Add papers to profile
    if papers_to_add:
        profile.past_papers.extend(papers_to_add)
        save_user_profile(profile, config.profile_path)
        logger.info(f"Added {len(papers_to_add)} papers to profile from reactions")

    return len(papers_to_add)


def run_bot(
        config_path: str = "config/config.json", 
        dry_run: bool = False, 
        use_scholar_inbox: bool = False
    ) -> None:
    if use_scholar_inbox:
        return run_bot_scholar_inbox(config_path, dry_run)
    else:
        return run_bot_ai(config_path, dry_run)

# TODO
def run_bot_scholar_inbox(config_path: str = "config/config.json", dry_run: bool = False) -> None:
    raise NotImplementedError("Scholar Inbox integration not yet implemented.")
    

def run_bot_ai(config_path: str = "config/config.json", dry_run: bool = False) -> None:
    """Run the Slack bot to fetch, filter, and post papers.

    Args:
        config_path: Path to configuration file.
        dry_run: If True, don't post to Slack, just show what would be posted.
    """
    # 1. Load config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    logger.info(f"Config: {config.categories}, last {config.days} day(s), top {config.top_k}")

    # 2. Fetch reactions and update profile before filtering (skip in dry run)
    if not dry_run:
        added_count = fetch_reactions_and_update_profile(config)
        if added_count > 0:
            logger.info(f"Added {added_count} papers to profile from reactions")

    # 3. Load user profile (after potential updates from reactions)
    logger.info(f"Loading profile from {config.profile_path}...")
    profile = load_user_profile(config.profile_path)
    logger.info(
        f"Profile '{profile.name}': {len(profile.topics)} topics, "
        f"{len(profile.keywords)} keywords, {len(profile.past_papers)} past papers"
    )

    # 4. Fetch papers from arXiv
    categories = [ArxivCategory(cat) for cat in config.categories]
    logger.info(f"Fetching papers from {config.categories} (last {config.days} day(s))...")
    papers = get_papers(
        categories=categories,
        days=config.days,
        max_results=-1,
        mode=OutputMode.TITLE_ABSTRACT,
    )

    if not papers:
        logger.warning("No papers found from arXiv")
        return

    logger.info(f"Fetched {len(papers)} papers")

    # 5. Filter by profile
    logger.info("Filtering papers by profile...")
    filtered = filter_papers(
        papers=papers,
        profile=profile,
        top_k=config.top_k * 2,  # Get extra in case some are already posted
        use_cache=True,
    )

    if not filtered:
        logger.warning("No papers matched the profile")
        return

    # 6. Remove already posted papers
    filtered = filter_already_posted(filtered)

    if not filtered:
        logger.info("All matching papers have already been posted")
        return

    # Limit to top_k
    papers_to_post = filtered[: config.top_k]
    logger.info(f"Selected {len(papers_to_post)} papers to post")

    # 7. Format and post (or dry run)
    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Would post the following:\n")
        for i, paper in enumerate(papers_to_post, 1):
            print(format_paper_message(paper, i))
            print()
        print("=" * 60)
    else:
        # Post to Slack (each paper as separate message)
        posted_results = post_to_slack(config, papers_to_post)

        if posted_results:
            # 8. Record posted papers with full metadata
            # Create a mapping from arxiv_id to message_ts
            ts_map = {arxiv_id: ts for arxiv_id, ts in posted_results}

            posted_papers = [
                {
                    "arxiv_id": p.paper.arxiv_id,
                    "message_ts": ts_map[p.paper.arxiv_id],
                    "title": p.paper.title,
                    "abstract": p.paper.abstract,
                }
                for p in papers_to_post
                if p.paper.arxiv_id in ts_map
            ]
            save_posted_papers(posted_papers)
            logger.info(f"Successfully posted {len(posted_results)} papers to Slack")
        else:
            logger.error("Failed to post any papers to Slack")


def main():
    """CLI entry point for the Slack bot."""
    parser = argparse.ArgumentParser(
        description="Post top ML papers to Slack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be posted
  python -m ai_pod.slack_bot --dry-run

  # Post to Slack
  python -m ai_pod.slack_bot

  # Use custom config
  python -m ai_pod.slack_bot -c config/custom_config.json
""",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.json",
        help="Path to configuration file (default: config/config.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't post to Slack, just show what would be posted",
    )

    args = parser.parse_args()

    try:
        run_bot(config_path=args.config, dry_run=args.dry_run)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
