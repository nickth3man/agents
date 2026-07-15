# OpenRouter Chat Agent — Python

Simple multi-turn terminal chat agent using OpenRouter's Chat Completions API.

## Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

## Setup

Edit `.env` at the repo root with your credentials:

```
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/auto
```

`OPENROUTER_MODEL` is optional (defaults to `openrouter/auto`).

## Run

```bash
cd python
python chat.py
```

Type `exit` or `quit` to end a conversation.
