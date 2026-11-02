// ── Provided helper: SHA-256 (Buffer in, Buffer out) ──
function sha256(data) {
    return require('crypto').createHash('sha256').update(Buffer.from(data)).digest();
}
