"""Slack bot for posting daily paper digests."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from ai_pod.filter_papers import filter_papers, load_user_profile
from ai_pod.get_papers import get_papers
from ai_pod.models import ArxivCategory, OutputMode
from ai_pod.posted_papers import (
    filter_already_posted,
    load_posted_papers,
    save_posted_papers,
)
from ai_pod.slack_utils import (
    fetch_reactions_and_update_profile,
    format_paper_message,
    import_papers_from_channel,
    post_summary_footer,
    post_to_slack,
)
from ai_pod.summary import generate_paper_summary


@dataclass
class SlackConfig:
    """Configuration for the Slack bot."""

    bot_token: str
    channel_id: str
    profile_path: str
    categories: list[str]
    days: int
    top_k: int
    openai_api_key: Optional[str] = None


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
        openai_api_key=data.get("openai_api_key"),
    )


def run_bot(config_path: str = "config/config.json", dry_run: bool = False, allow_duplication: bool = False) -> None:
    """Run the Slack bot to fetch, filter, and post papers.

    Args:
        config_path: Path to configuration file.
        dry_run: If True, don't post to Slack, just show what would be posted.
        allow_duplication: If True, allow duplication of papers in the digest.
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
    if not allow_duplication:
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

        # Generate and display summary if API key is configured
        if config.openai_api_key:
            previous_papers = load_posted_papers()
            logger.info("Generating paper summary...")
            summary = generate_paper_summary(papers_to_post, previous_papers, config.openai_api_key)
            if summary:
                print("-" * 60)
                print("TODAY'S DIGEST SUMMARY:\n")
                print(summary)
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

            # 9. Generate and post summary footer if API key is configured
            if config.openai_api_key:
                previous_papers = load_posted_papers()
                logger.info("Generating paper summary...")
                summary = generate_paper_summary(papers_to_post, previous_papers, config.openai_api_key)
                if summary:
                    post_summary_footer(config, summary)
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

  # Import existing papers from Slack channel
  python -m ai_pod.slack_bot --import-from-channel

  # Import from a different channel
  python -m ai_pod.slack_bot --import-from-channel --import-channel C01234ABCDE

  # Import last 30 days without fetching metadata
  python -m ai_pod.slack_bot --import-from-channel --import-days 30 --no-fetch-metadata
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
    parser.add_argument(
        "--allow-duplication",
        action="store_true",
        help="Allow duplication of papers in the digest",
    )
    parser.add_argument(
        "--import-from-channel",
        action="store_true",
        help="Import existing papers from Slack channel history",
    )
    parser.add_argument(
        "--import-channel",
        type=str,
        default=None,
        help="Channel ID to import from (overrides config channel_id)",
    )
    parser.add_argument(
        "--import-days",
        type=int,
        default=None,
        help="Number of days of history to import (default: all history)",
    )
    parser.add_argument(
        "--no-fetch-metadata",
        action="store_true",
        help="Don't fetch paper metadata from arXiv when importing",
    )
    args = parser.parse_args()

    try:
        # Import mode: scan channel and add papers to tracking
        if args.import_from_channel:
            config = load_config(args.config)
            count = import_papers_from_channel(
                config=config,
                days=args.import_days,
                fetch_metadata=not args.no_fetch_metadata,
                channel_id=args.import_channel,
            )
            logger.info(f"Import complete: {count} papers added to tracking")
        else:
            # Normal mode: fetch, filter, and post papers
            run_bot(config_path=args.config, dry_run=args.dry_run, allow_duplication=args.allow_duplication)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
