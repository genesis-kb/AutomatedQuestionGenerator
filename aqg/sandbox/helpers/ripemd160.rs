// ── Provided helper: RIPEMD-160 (std only) ──
#[allow(dead_code)]
fn ripemd160(data: &[u8]) -> Vec<u8> {
    const R: [usize; 80] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
    ];
    const RP: [usize; 80] = [
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
    ];
    const S: [u32; 80] = [
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
    ];
    const SP: [u32; 80] = [
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
    ];
    const K: [u32; 5] = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E];
    const KP: [u32; 5] = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000];
    fn f(j: usize, x: u32, y: u32, z: u32) -> u32 {
        match j {
            0..=15 => x ^ y ^ z,
            16..=31 => (x & y) | (!x & z),
            32..=47 => (x | !y) ^ z,
            48..=63 => (x & z) | (y & !z),
            _ => x ^ (y | !z),
        }
    }
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    for chunk in msg.chunks(64) {
        let mut x = [0u32; 16];
        for i in 0..16 {
            x[i] = u32::from_le_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        let (mut ap, mut bp, mut cp, mut dp, mut ep) = (h[0], h[1], h[2], h[3], h[4]);
        for j in 0..80 {
            let t = a
                .wrapping_add(f(j, b, c, d))
                .wrapping_add(x[R[j]])
                .wrapping_add(K[j / 16])
                .rotate_left(S[j])
                .wrapping_add(e);
            a = e;
            e = d;
            d = c.rotate_left(10);
            c = b;
            b = t;
            let t = ap
                .wrapping_add(f(79 - j, bp, cp, dp))
                .wrapping_add(x[RP[j]])
                .wrapping_add(KP[j / 16])
                .rotate_left(SP[j])
                .wrapping_add(ep);
            ap = ep;
            ep = dp;
            dp = cp.rotate_left(10);
            cp = bp;
            bp = t;
        }
        let t = h[1].wrapping_add(c).wrapping_add(dp);
        h[1] = h[2].wrapping_add(d).wrapping_add(ep);
        h[2] = h[3].wrapping_add(e).wrapping_add(ap);
        h[3] = h[4].wrapping_add(a).wrapping_add(bp);
        h[4] = h[0].wrapping_add(b).wrapping_add(cp);
        h[0] = t;
    }
    h.iter().flat_map(|v| v.to_le_bytes()).collect()
}
