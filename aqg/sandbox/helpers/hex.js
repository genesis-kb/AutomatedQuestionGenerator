// ── Provided helper: hex encode / decode ──
function hexDecode(s) {
    return Buffer.from(s, 'hex');
}

function hexEncode(data) {
    return Buffer.from(data).toString('hex');
}
