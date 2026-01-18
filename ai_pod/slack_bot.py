"""Slack bot for posting daily paper digests."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from ai_pod.filter_papers import filter_papers, load_user_profile
from ai_pod.get_papers import get_papers
from ai_pod.models import ArxivCategory, FilteredPaper, OutputMode


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


def load_posted_papers() -> set[str]:
    """Load set of previously posted paper arxiv_ids.

    Returns:
        Set of arxiv_ids that have been posted.
    """
    if not POSTED_PAPERS_PATH.exists():
        return set()

    with open(POSTED_PAPERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return set(data.get("posted_ids", []))


def save_posted_papers(arxiv_ids: list[str]) -> None:
    """Save newly posted paper IDs to the tracking file.

    Args:
        arxiv_ids: List of arxiv_ids that were just posted.
    """
    POSTED_PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_posted_papers()
    existing.update(arxiv_ids)

    with open(POSTED_PAPERS_PATH, "w", encoding="utf-8") as f:
        json.dump({"posted_ids": list(existing)}, f, indent=2)

    logger.info(f"Saved {len(arxiv_ids)} new paper IDs to posted papers tracking")


def filter_already_posted(papers: list[FilteredPaper]) -> list[FilteredPaper]:
    """Remove papers that have already been posted.

    Args:
        papers: List of filtered papers.

    Returns:
        Papers that haven't been posted yet.
    """
    posted = load_posted_papers()
    filtered = [p for p in papers if p.paper.arxiv_id not in posted]

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


def format_daily_digest(papers: list[FilteredPaper]) -> list[dict]:
    """Format papers as Slack blocks for rich formatting.

    Args:
        papers: List of filtered papers to include.

    Returns:
        List of Slack block elements.
    """
    today = datetime.now().strftime("%B %d, %Y")

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Top {len(papers)} Papers for {today}",
                "emoji": True,
            },
        },
        {"type": "divider"},
    ]

    # Add each paper
    paper_texts = []
    for i, paper in enumerate(papers, 1):
        paper_texts.append(format_paper_message(paper, i))

    # Combine papers into sections (Slack has block limits)
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n\n".join(paper_texts),
            },
        }
    )

    return blocks


def post_to_slack(config: SlackConfig, papers: list[FilteredPaper]) -> bool:
    """Post formatted message to Slack channel.

    Args:
        config: Slack configuration.
        papers: List of papers to post.

    Returns:
        True if successful, False otherwise.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=config.bot_token)

    blocks = format_daily_digest(papers)

    try:
        response = client.chat_postMessage(
            channel=config.channel_id,
            blocks=blocks,
            text=f"Top {len(papers)} ML papers for today",  # Fallback for notifications
        )
        logger.info(f"Posted to Slack channel {config.channel_id}: {response['ts']}")
        return True

    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        return False


def run_bot(config_path: str = "config/config.json", dry_run: bool = False) -> None:
    """Run the Slack bot to fetch, filter, and post papers.

    Args:
        config_path: Path to configuration file.
        dry_run: If True, don't post to Slack, just show what would be posted.
    """
    # 1. Load config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    logger.info(f"Config: {config.categories}, last {config.days} day(s), top {config.top_k}")

    # 2. Load user profile
    logger.info(f"Loading profile from {config.profile_path}...")
    profile = load_user_profile(config.profile_path)
    logger.info(
        f"Profile '{profile.name}': {len(profile.topics)} topics, "
        f"{len(profile.keywords)} keywords, {len(profile.past_papers)} past papers"
    )

    # 3. Fetch papers from arXiv
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

    # 4. Filter by profile
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

    # 5. Remove already posted papers
    filtered = filter_already_posted(filtered)

    if not filtered:
        logger.info("All matching papers have already been posted")
        return

    # Limit to top_k
    papers_to_post = filtered[: config.top_k]
    logger.info(f"Selected {len(papers_to_post)} papers to post")

    # 6. Format and post (or dry run)
    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Would post the following:\n")
        for i, paper in enumerate(papers_to_post, 1):
            print(format_paper_message(paper, i))
            print()
        print("=" * 60)
    else:
        # Post to Slack
        success = post_to_slack(config, papers_to_post)

        if success:
            # 7. Record posted papers
            posted_ids = [p.paper.arxiv_id for p in papers_to_post]
            save_posted_papers(posted_ids)
            logger.info(f"Successfully posted {len(papers_to_post)} papers to Slack")
        else:
            logger.error("Failed to post to Slack")


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
