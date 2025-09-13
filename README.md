# Telegram Media Downloader

![Banner](https://i.imgur.com/JwZ1VjP.png)

# 📖 Description
- A utility to download all media from a Telegram chat/channel and organize files by type (audio, videos, photos, documents, stickers, animations, other).
- Supports robust resume, queue-based runs (scan once, download many), atomic writes, and structured logging.
- Handles Telegram FloodWait automatically with session rotation using multiple accounts placed in `accs/`.

# ⚙️ Project Setup Guide

Welcome to the setup guide! Follow these steps to configure and run the downloader.

## Getting Started

### 1. Prepare Telegram API access

1. Create a Telegram application at https://my.telegram.org/apps and obtain `API_ID` and `API_HASH`.
2. Ensure the account(s) used are members of the target chat/channel.
3. Place Pyrogram session files (after logging in once) into `accs/` for rotation, or set a single `SESSION_NAME`.

### 2. Configure Environment Variables

Create a `.env` file (or use `.env.example`) with the following keys:

```env
API_ID=your_api_id
API_HASH=your_api_hash
CHAT_ID=-1001234567890                 # target chat/channel id
SESSION_NAME=mainsession               # used if accs/ is empty

# Paths
DOWNLOADS_DIR=downloads
STATE_FILE=downloads/.downloaded_ids.txt
QUEUE_FILE=downloads/.queue_ids.txt

# Logging
LOG_DIR=downloads/logs
LOG_LEVEL=INFO

# Rotation and limits
ACCS_DIR=accs
FLOODWAIT_SWITCH_SECS=120
DOWNLOAD_DELAY=1.5

# Connection tuning
CONNECT_TIMEOUT=30
CONNECT_RETRIES=3
CONNECT_RETRY_DELAY=10

# Optional
LIMIT=0                               # used only with --scan (0 = all)
MAX_RETRIES=5
```

## Setup Instructions

1. Create and activate a virtual environment, then install requirements:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place account sessions in `accs/` (e.g., `+1234567890.session`). If you use a single session, keep it in the project root and set `SESSION_NAME`.

3. Populate the download queue (one-time or as needed):

```powershell
py .\download.py --scan
```

4. Start downloading (will rotate sessions and resume safely):

```powershell
py .\download.py
```

## Notes

- `.gitignore` excludes `.env`, `accs/*.session`, logs, and downloads to avoid leaking secrets or pushing large files.
- The downloader writes temporary `.part` files and renames atomically when complete.
- On FloodWait or RPC errors, it rotates sessions or waits the minimal cooldown and resumes.
- Use `LIMIT` during `--scan` to bound initial queueing if desired.

# 🚀 Usage

- `--scan`: Scan chat history and append media IDs to the queue file. Does not download.
- Default run: Loads the queue and downloads sequentially with retries and integrity checks.

Happy downloading!
