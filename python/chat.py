import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

def _load_dotenv(path: str | Path) -> None:
    """Load a .env file, skipping blank lines and # comments.
    Does NOT override environment variables already set."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass  # .env file is optional


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
API_KEY = os.getenv("OPENROUTER_API_KEY")


def complete(messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"model": MODEL, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")

    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"OpenRouter returned unexpected content: {data}")

    return content


def main() -> int:
    if not API_KEY:
        print("Set OPENROUTER_API_KEY before running this program.", file=sys.stderr)
        return 1

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful, concise assistant."}
    ]

    print(f"OpenRouter chat agent using {MODEL}. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        try:
            answer = complete(messages)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(f"Assistant: {answer}\n")
        messages.append({"role": "assistant", "content": answer})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
