# ── Provided helper: hex encode / decode ──
def hex_decode(s: str) -> bytes:
    return bytes.fromhex(s)


def hex_encode(data: bytes) -> str:
    return bytes(data).hex()
