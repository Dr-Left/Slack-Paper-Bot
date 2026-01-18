# AI Coffee Time

AI-powered tool that generates personalized content for ML researchers. Curated, highly-personalized, and intriguing.

## Features

- **Paper Fetching**: Fetch recent papers from arXiv by category
- **Paper Filtering**: SPECTER2-based semantic filtering by user interests
- **Slack Bot**: Daily digest of top papers posted to Slack

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e .
```

## Slack Bot Setup

The Slack bot posts daily paper digests to a channel based on your research interests.

### 1. Create a Slack App

1. Go to https://api.slack.com/apps
2. Click **Create New App** → **From scratch**
3. Name it (e.g., "AI Paper Bot") and select your workspace
4. Go to **OAuth & Permissions** in the sidebar
5. Under **Bot Token Scopes**, add:
   - `chat:write` - Post messages
   - `chat:write.public` - Post to public channels without joining
6. Click **Install to Workspace**
7. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### 2. Get Channel ID

- Right-click the target channel in Slack
- Click "View channel details"
- Scroll down and copy the **Channel ID** (starts with `C`)

### 3. Configure the Bot

```bash
# Copy the example config
cp config/config.example.json config/config.json

# Edit with your values
```

Edit `config/config.json`:
```json
{
  "bot_token": "xoxb-your-token-here",
  "channel_id": "C0123456789",
  "profile_path": "profiles/efficient_ml.json",
  "categories": ["cs.LG", "cs.AI", "cs.CL"],
  "days": 1,
  "top_k": 5
}
```

### 4. Test the Bot

```bash
# Dry run (preview without posting)
python -m ai_pod.slack_bot --dry-run

# Post to Slack
python -m ai_pod.slack_bot
```

### 5. Schedule Daily Runs (Optional)

```bash
# Edit crontab
crontab -e

# Add daily 8am job
0 8 * * * cd /path/to/ai-pod && .venv/bin/python -m ai_pod.slack_bot >> logs/slack_bot.log 2>&1
```

### Customizing Your Profile

Edit `profiles/efficient_ml.json` or create a new profile with your interests:

- **topics**: Natural language descriptions of research areas
- **keywords**: Specific terms to boost
- **past_papers**: Papers you found interesting (title + abstract)

## CLI Commands

```bash
# Fetch papers from arXiv
python -m ai_pod.get_papers -c cs.LG cs.AI -d 7 -n 20

# Filter papers by profile
python -m ai_pod.filter_papers -p profiles/efficient_ml.json --fetch -c cs.LG -d 3 -t 0.3

# Run Slack bot
python -m ai_pod.slack_bot --dry-run
```

## Technical Details

### Paper Fetching

Uses [arXiv API](https://info.arxiv.org/help/api/basics.html#quickstart) to fetch recent papers.

### Paper Filtering

Uses SPECTER2 embeddings for semantic similarity matching