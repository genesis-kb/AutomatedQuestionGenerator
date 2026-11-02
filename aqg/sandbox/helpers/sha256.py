# ── Provided helper: SHA-256 ──
import hashlib as _hashlib


def sha256(data: bytes) -> bytes:
    return _hashlib.sha256(bytes(data)).digest()
