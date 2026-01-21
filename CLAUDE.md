# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

AI Coffee Time is an AI-powered tool that generates personalized podcast content for machine learning researchers. The content is designed to be consumed during morning coffee time—curated, highly personalized, and intriguing.

## Tech Stack

- **Language**: Python 3.13+
- **Environment**: Virtual environment (`.venv`)
- **ML**: SPECTER2 (allenai/specter2_base) for paper embeddings
- **Dependencies**: transformers, adapters, torch, numpy, loguru, slack_sdk

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install package in development mode
pip install -e .

# Fetch papers from arXiv
python -m ai_pod.get_papers -c cs.LG cs.AI -d 7 -n 20 --show-affiliations

# Filter papers by user profile
python -m ai_pod.filter_papers -p profiles/example_profile.json --fetch -c cs.LG -d 3 -t 0.3

# Run Slack bot (dry run)
python -m ai_pod.slack_bot --dry-run

# Run Slack bot (post to Slack)
python -m ai_pod.slack_bot

# Import existing papers from Slack channel
python -m ai_pod.slack_bot --import-from-channel

# Import from a different channel
python -m ai_pod.slack_bot --import-from-channel --import-channel C01234ABCDE

# Import last 30 days without fetching metadata (faster)
python -m ai_pod.slack_bot --import-from-channel --import-days 30 --no-fetch-metadata

# Run tests (when implemented)
pytest

# Type checking (when implemented)
mypy .
```

## Project Structure

```
ai-pod/
├── ai_pod/
│   ├── __init__.py
│   ├── models.py            # Data models (Paper, Author, UserProfile, etc.)
│   ├── get_papers.py        # arXiv API paper fetching
│   ├── filter_papers.py     # SPECTER2-based semantic filtering
│   ├── slack_bot.py         # Slack bot main entry point
│   ├── slack_utils.py       # Slack formatting, posting, reactions, import
│   ├── posted_papers.py     # Posted papers tracking utilities
│   ├── summary.py           # LLM summary generation
│   └── utils/
│       ├── cache_utils.py       # Paper caching utilities
│       └── embedding_cache.py   # Embedding caching utilities
├── config/
│   ├── config.example.json  # Slack bot config template
│   └── config.json          # Actual config (gitignored)
├── profiles/
│   ├── example_profile.json # Example user profile
│   └── efficient_ml.json    # Efficient ML research profile
├── data/                    # Cache directory (auto-created)
│   ├── posted_papers.json   # Tracking of posted papers with metadata
│   └── paper_embeddings.json # Cached SPECTER2 embeddings
├── logs/                    # Log files (auto-created, gitignored)
├── setup.py
├── CLAUDE.md
└── README.md
```

## Implemented Components

### Paper Fetching (`get_papers.py`)
- Fetches papers from arXiv API by category and date range
- Extracts author affiliations when available (via `<arxiv:affiliation>`)
- Caches results to `data/papers_*.json` with configurable TTL
- CLI: `python -m ai_pod.get_papers --help`

### Paper Filtering (`filter_papers.py`)
- Uses SPECTER2 embeddings for semantic similarity
- Filters papers based on user profile (topics, keywords, past papers)
- Weighted similarity: topics (40%), keywords (30%), past papers (30%)
- Auto-detects device (CUDA > MPS > CPU)
- Caches embeddings to `data/paper_embeddings.json` and `data/profile_embeddings_*.json`
- CLI: `python -m ai_pod.filter_papers --help`

### Data Models (`models.py`)
- `Author`: name + optional affiliation
- `Paper`: arXiv paper with authors, abstract, categories
- `UserProfile`: researcher interests (topics, keywords, past papers)
- `FilteredPaper`: paper with similarity score

### Slack Bot (`slack_bot.py`, `slack_utils.py`, `posted_papers.py`, `summary.py`)
**Main Bot (`slack_bot.py`)**:
- Posts daily paper digests to Slack channel
- Uses `filter_papers.py` for semantic matching against user profile
- Tracks posted papers in `data/posted_papers.json` for deduplication
- Configurable via `config/config.json` (bot token, channel, categories, etc.)
- Requires Slack App with `chat:write`, `chat:write.public`, and `reactions:read` scopes

**Slack Utilities (`slack_utils.py`)**:
- Formatting functions for Slack blocks and messages
- `post_to_slack()` - Posts papers as separate messages
- `fetch_reactions_and_update_profile()` - Reads 🔥 reactions and adds papers to profile
- `import_papers_from_channel()` - Scans channel history for arXiv papers
- Multi-layer fallback for metadata fetching (API → arxiv-txt.org → HTML scraping)

**Posted Papers Tracking (`posted_papers.py`)**:
- `load_posted_papers()` - Loads tracking data with backward compatibility
- `save_posted_papers()` - Saves with full metadata (arxiv_id, message_ts, title, abstract)
- `filter_already_posted()` - Removes duplicates before posting

**Summary Generation (`summary.py`)**:
- `generate_paper_summary()` - OpenAI GPT-4o summaries of daily digest
- `find_similar_past_paper()` - Uses SPECTER2 to find similar previously posted papers
- Generates contrastive summaries highlighting differences from past papers

**Key Features**:
- **Channel Import**: Bootstrap from existing channel history (`--import-from-channel`)
- **Reaction Learning**: React with 🔥 to papers you like, automatically added to profile
- **Rate Limiting**: Exponential backoff and fallback sources for arXiv API
- **Deduplication**: Prevents reposting papers across bot runs
- **LLM Summaries**: Optional GPT-4o summaries with contrastive analysis

## Architecture Notes

Key components still to be built:
- Text-to-speech synthesis for podcast generation
- Content summarization and script writing
- Email digest option (alternative to Slack)

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Prefer dataclasses or Pydantic models for data structures
- Write docstrings for public functions and classes
