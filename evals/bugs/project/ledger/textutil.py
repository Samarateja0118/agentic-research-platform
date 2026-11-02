import re


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return s.strip("-")


def truncate(text: str, width: int) -> str:
    """Shorten to at most `width` characters, ending with '...' when cut."""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."
