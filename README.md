# OpenRouter Chat Agents in Five Languages

Minimal multi-turn terminal chat agents using OpenRouter's Chat Completions API.

## Implementations

| Language | Runtime | Directory | README |
|----------|---------|-----------|--------|
| Python | Python 3.10+ | [python/](python/) | [README](python/README.md) |
| Go | Go 1.22+ | [go/](go/) | [README](go/README.md) |
| TypeScript | Bun 1.0+ | [bun/](bun/) | [README](bun/README.md) |
| C# | .NET 8.0+ | [csharp/](csharp/) | [README](csharp/README.md) |
| Rust | Rust 1.75+ | [rust/](rust/) | [README](rust/README.md) |

## Configuration

Edit `.env` at the repository root with your OpenRouter credentials:

```
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/auto
```

`OPENROUTER_MODEL` is optional (defaults to `openrouter/auto`).

Each agent automatically loads `.env` on startup. Environment variables already set in your shell take precedence over `.env` values.
