import re

def is_valid_username(username: str) -> bool:
    if not username or len(username) < 3 or len(username) > 32:
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", username))

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    # Strip dangerous HTML or control characters
    clean = re.sub(r"<[^>]*>", "", text)
    return clean.strip()
