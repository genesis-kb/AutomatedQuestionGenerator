from aqg.textutil import sha256_hex, slugify


def test_slugify_parity_with_hello_bitcoin():
    assert slugify("Encode a CompactSize Integer") == "encode-a-compactsize-integer"
    assert slugify("  Hello, World!! ") == "hello-world"
    assert slugify("BIP-340 Schnorr: Verify") == "bip-340-schnorr-verify"
    assert slugify("Ünïcode — dash") == "n-code-dash"


def test_sha256_hex():
    assert sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
