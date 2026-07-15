# OpenRouter Chat Agent — C\#

Simple multi-turn terminal chat agent using OpenRouter's Chat Completions API.

## Prerequisites

- .NET 8.0+ SDK
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
cd csharp
dotnet run
```

Type `exit` or `quit` to end a conversation.
