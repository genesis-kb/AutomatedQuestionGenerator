"""Small text helpers shared across stages."""

from __future__ import annotations

import hashlib
import re


def slugify(text: str) -> str:
    """Identical to hello-bitcoin's admin `generate_slug` (backend/routes/admin.py)."""
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
