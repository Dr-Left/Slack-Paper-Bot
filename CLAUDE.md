# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

AI Coffee Time is an AI-powered tool that generates personalized podcast content for machine learning researchers. The content is designed to be consumed during morning coffee time—curated, highly personalized, and intriguing.

## Tech Stack

- **Language**: Python 3.13+
- **Environment**: Virtual environment (`.venv`)
- **ML**: SPECTER2 (allenai/specter2_base) for paper embeddings
- **Dependencies**: transformers, adapters, torch, numpy, loguru

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
│   └── utils/
│       ├── cache_utils.py       # Paper caching utilities
│       └── embedding_cache.py   # Embedding caching utilities
├── profiles/
│   └── example_profile.json # Example user profile
├── data/                    # Cache directory (auto-created)
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

## Architecture Notes

Key components still to be built:
- Text-to-speech synthesis for podcast generation
- Content summarization and script writing

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Prefer dataclasses or Pydantic models for data structures
- Write docstrings for public functions and classes
