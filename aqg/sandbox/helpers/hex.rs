// ── Provided helper: hex encode / decode (std only) ──
#[allow(dead_code)]
fn hex_decode(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    assert!(bytes.len() % 2 == 0, "hex string must have even length");
    (0..bytes.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("invalid hex"))
        .collect()
}

#[allow(dead_code)]
fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}
