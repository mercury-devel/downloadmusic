# PhotoEditor Telegram

![Баннер](https://i.imgur.com/LKVVDhb.jpeg)

# 📖 Description
- A bot that allows you to add different effects to photos such as brightness, noise, moire, quality distortion, etc.
- Allows you to add text to photos
- Also supports AI with picture generation, illusion effect and photo cropping

# ⚙️ Project Setup Guide

Welcome to the project setup guide! Follow these steps to configure and run the project, whether you're using Python or Docker.

## Getting Started

### 1. Create and Configure Your Telegram Channel

1. **Create a Telegram Channel:** 
   You need a Telegram channel to make your bot popular. People can use your bot by joining this channel.

2. **Add Your Bot to the Channel:**
   - Create a Telegram bot.
   - Add the bot to your newly created channel.


### 2. Configure Environment Variables

Add the following information to your `.env` file:

```env
API_TOKEN=your_telegram_bot_token
CHANNEL=your_channel_id (ex. -100123123123) 
LOG_CHAT=chat or channel where you can moderate content that users upload into your bot (ex. -100123123123)
CHANNEL_LINK=your_channel_link (ex. t.me/***)
DB_PATH=db.db path to your database
ADMIN_IDS=[123, 456](where are 123 and 456 - telegram ids of admins)
```

## Setup Instructions

1. **Install Docker**

   Follow the instructions on [Docker's official installation guide](https://docs.docker.com/engine/install/).

2. **Run Docker Setup Menu**

   Execute the following command to start the Docker setup menu:

   ```bash
   sh run.sh
   ```

   Using Docker, you can:
   - Build the project
   - Create and start a container
   - Stop the container
   - Restart the container
   - Remove the container
   - Show container logs

## Notes

- Ensure that your `.env` file is properly configured with the correct values.
- Follow Docker's documentation for any additional configuration or troubleshooting.

# 🚀 Usage

- /start - start using bot
- /edit - photo editor
- /ai - artificial intellingense for work with photo
