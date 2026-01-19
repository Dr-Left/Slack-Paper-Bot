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
   - Add bot scopes: `chat:write`, `chat:write.public`
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

Papers you react to with `:fire:` in Slack are automatically added to your profile’s `past_papers`.

### Core commands

```bash
# Fetch papers from arXiv (shows affiliations when present)
python -m ai_pod.get_papers -c cs.LG cs.AI -d 7 -n 20 --show-affiliations

# Filter papers by profile (fetching from arXiv)
python -m ai_pod.filter_papers -p profiles/example_profile.json --fetch -c cs.LG -d 3 -t 0.3

# Filter using an existing cached papers file
python -m ai_pod.filter_papers -p profiles/example_profile.json --papers-cache data/papers_*.json

# Run Slack bot
python -m ai_pod.slack_bot --dry-run
python -m ai_pod.slack_bot

# Import historic arXiv links from a channel into tracking
python -m ai_pod.slack_bot --import-from-channel
python -m ai_pod.slack_bot --import-from-channel --import-channel C01234ABCDE --import-days 30 --no-fetch-metadata
```

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

### Architecture (high level)

- `ai_pod.get_papers`: arXiv fetching + XML parsing + caching.
- `ai_pod.filter_papers`: SPECTER2 model loading, profile & paper embeddings, similarity scoring.
- `ai_pod.slack_bot`: Orchestrates fetch → filter → dedupe → post, plus optional summary.
- `ai_pod.slack_utils`: Slack formatting, posting, reaction-based profile updates, channel import.
- `ai_pod.summary`: GPT-4o-based digest summarization using OpenAI Python SDK.

## License
MIT License.
