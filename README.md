# Slack AI Paper Bot

AI-powered tool that curates personalized ML paper digests (and future podcast-style content) for researchers.

### Features

- **Paper fetching**: Fetch recent papers from arXiv by category (with optional affiliations).
- **Semantic filtering**: SPECTER2-based matching against a user profile (topics, keywords, past papers).
- **Slack digests**: Daily top papers posted to a Slack channel, with reaction-based profile updates.
- **LLM summary**: Optional GPT-4o-powered digest summary footer for Slack.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.13+ is recommended.

### Configure Slack bot

1. **Create Slack app**
   - Go to `https://api.slack.com/apps`
   - Create a new app from scratch
   - Add bot scopes: `chat:write`, `chat:write.public`, `reactions:read`
   - Install the app and copy the **Bot User OAuth Token** (`xoxb-...`)
2. **Find channel ID**
   - Open channel details in Slack and copy the **Channel ID** (`C...`)
3. **Create config**

```bash
cp config/config.example.json config/config.json
```

Edit `config/config.json`:

```json
{
  "bot_token": "xoxb-your-token-here",
  "channel_id": "C0123456789",
  "profile_path": "profiles/efficient_ml.json",
  "categories": ["cs.LG", "cs.AI", "cs.CL"],
  "days": 1,
  "top_k": 5,
  "openai_api_key": "sk-... (optional for summaries)"
}
```

### Profiles

Profiles live in `profiles/*.json`. See `profiles/example_profile.json` or `profiles/efficient_ml.json`:

- **name**: Profile name.
- **topics**: Free-text research interests.
- **keywords**: Short key phrases to emphasize.
- **past_papers**: Optional list of `{title, abstract, arxiv_id}`.
- **preferred_authors**: Optional list of author names.

**Reaction Learning**: Papers you react to with `:fire:` (🔥) in Slack are automatically added to your profile's `past_papers`. The bot checks for reactions before each run and updates your profile to improve future recommendations.

### Core commands

```bash
# Fetch papers from arXiv (shows affiliations when present)
python -m ai_pod.get_papers -c cs.LG cs.AI -d 7 -n 20 --show-affiliations

# Filter papers by profile (fetching from arXiv)
python -m ai_pod.filter_papers -p profiles/example_profile.json --fetch -c cs.LG -d 3 -t 0.3

# Filter using an existing cached papers file
python -m ai_pod.filter_papers -p profiles/example_profile.json --papers-cache data/papers_*.json

# Run Slack bot (dry run to preview)
python -m ai_pod.slack_bot --dry-run

# Run Slack bot (post to Slack)
python -m ai_pod.slack_bot

# Allow duplicate papers in digest (useful for testing)
python -m ai_pod.slack_bot --allow-duplication
```

### Channel import (bootstrapping)

If you've been posting papers manually or want to import from an existing channel:

```bash
# Import all papers from configured channel
python -m ai_pod.slack_bot --import-from-channel

# Import from a different channel
python -m ai_pod.slack_bot --import-from-channel --import-channel C01234ABCDE

# Import only last 30 days
python -m ai_pod.slack_bot --import-from-channel --import-days 30

# Fast import without fetching metadata from arXiv
python -m ai_pod.slack_bot --import-from-channel --import-days 30 --no-fetch-metadata
```

**How it works:**
- Scans Slack channel messages for `arxiv.org/abs/` links
- Extracts arXiv IDs and fetches metadata (title, abstract) from arXiv API
- Adds papers to `data/posted_papers.json` to prevent duplicates
- Uses fallback methods (arxiv-txt.org, HTML scraping) if API rate-limited
- Rate-limited to 5 seconds between requests to respect arXiv's limits

### LLM summaries (optional)

Add your OpenAI API key to `config/config.json` to enable GPT-4o summaries:

```json
{
  "openai_api_key": "sk-..."
}
```

The bot will generate a digest summary footer with:
- Bullet points for each recommended paper
- Contrastive analysis against similar previously posted papers
- Links to similar past papers for context
- Slack-friendly formatting with clickable arXiv links

### Scheduling (optional)

```bash
crontab -e

# Daily 8am digest
0 8 * * * cd /path/to/ai-pod && .venv/bin/python -m ai_pod.slack_bot >> logs/slack_bot.log 2>&1
```

### Data & caching

- **`data/posted_papers.json`**: Papers already posted to Slack (with timestamps, titles, abstracts).
- **`data/paper_embeddings.json`**: Cached SPECTER2 embeddings for papers.
- **`data/profile_embeddings_*.json`**: Cached profile embeddings by profile name.

Caches are used automatically and refreshed as needed.

### Troubleshooting

**Rate Limiting (HTTP 429 errors)**
The bot uses multiple fallback methods for fetching paper metadata:
1. Official arXiv API (primary, with retry logic)
2. arxiv-txt.org (fallback #1)
3. arXiv HTML scraping (fallback #2)

If you hit rate limits during channel import, the bot will:
- Wait and retry with exponential backoff (5s, 10s, 15s)
- Automatically try fallback sources
- Continue processing remaining papers

For large imports, consider using `--no-fetch-metadata` for a faster initial import.

**Missing Papers in Digest**
- Check `data/posted_papers.json` - papers already posted won't appear again
- Use `--allow-duplication` flag for testing
- Adjust `top_k` in config to see more papers

**Profile Not Improving**
- Make sure `reactions:read` scope is enabled in your Slack app
- React with 🔥 to papers you find interesting
- Check logs to see if reactions are being detected

### Architecture (high level)

- `ai_pod.get_papers`: arXiv fetching + XML parsing + caching.
- `ai_pod.filter_papers`: SPECTER2 model loading, profile & paper embeddings, similarity scoring.
- `ai_pod.slack_bot`: Orchestrates fetch → filter → dedupe → post, plus optional summary.
- `ai_pod.slack_utils`: Slack formatting, posting, reaction-based profile updates, channel import.
- `ai_pod.posted_papers`: Tracking of posted papers with metadata for deduplication.
- `ai_pod.summary`: GPT-4o-based digest summarization using OpenAI Python SDK with contrastive analysis.