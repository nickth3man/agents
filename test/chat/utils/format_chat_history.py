def format_chat_history(history):
    """
    Format the chat history for LLM

    Args:
        history (list): The chat history list, each element contains role and content

    Returns:
        str: The formatted chat history string
    """
    if not history:
        return "No history"

    formatted_history = []
    for message in history:
        role = "user" if message.get("role") == "user" else "assistant"
        raw = message.get("content", "")
        # Gradio's messages format can deliver content as a string, a list of
        # parts (multimodal), or a dict. Normalize to a flat string.
        if isinstance(raw, list):
            content = "".join(
                part if isinstance(part, str) else str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in raw
            )
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw)

        # filter out thinking / deliberation placeholders
        if role == "assistant":
            if (
                content.startswith("- 🤔")
                or content.startswith("- ➡️")
                or content.startswith("- ⬅️")
                or content.startswith("_Deliberating")
                or content.startswith("_The floor is open")
                or content.startswith("_The bench is empty")
            ):
                continue
        formatted_history.append(f"{role}: {content}")

    return "\n".join(formatted_history) if formatted_history else "No history"
