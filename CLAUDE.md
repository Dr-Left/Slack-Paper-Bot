# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

AI Coffee Time is an AI-powered tool that generates personalized podcast content for machine learning researchers. The content is designed to be consumed during morning coffee time—curated, highly personalized, and intriguing.

## Tech Stack

- **Language**: Python 3.13+
- **Environment**: Virtual environment (`.venv`)

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Run tests (when implemented)
pytest

# Type checking (when implemented)
mypy .

# Linting (when implemented)
ruff check .
```

## Project Structure

```
ai-pod/
├── .venv/           # Python virtual environment
├── .gitignore       # Git ignore rules
├── CLAUDE.md        # This file
└── README.md        # Project documentation
```

## Architecture Notes

This is an MVP project. Key components to be built:
- Paper fetching/curation from sources (arXiv, etc.)
- Personalization engine based on researcher interests
- Text-to-speech synthesis for podcast generation
- Content summarization and script writing

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Prefer dataclasses or Pydantic models for data structures
- Write docstrings for public functions and classes
