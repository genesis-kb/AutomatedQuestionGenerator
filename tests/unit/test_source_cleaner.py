from aqg.source.cleaner import clean_text


def test_markers_and_timestamps_removed():
    raw = "[Music] Welcome everyone (applause)\n00:01:23 So today [00:01:23.456] we talk [laughter] about blocks."
    assert clean_text(raw) == "Welcome everyone\nSo today we talk about blocks."


def test_fillers_removed_without_breaking_punctuation():
    assert clean_text("So, um, the txid is, uh, reversed.") == "So, the txid is, reversed."
    assert clean_text("Um, so we hash it uh.") == "so we hash it."
    assert clean_text("The drum and the umbrella, erm, are fine") == "The drum and the umbrella, are fine"
    assert clean_text("UHM yes") == "yes"


def test_whitespace_and_paragraphs():
    raw = "First line   with   spaces.\r\n\r\n\r\n\r\nSecond\tparagraph.\n"
    assert clean_text(raw) == "First line with spaces.\n\nSecond paragraph."


def test_deterministic_and_empty():
    raw = " [MUSIC] uh hello (Applause) [01:23] world\n01:24 again "
    assert clean_text(raw) == clean_text(raw) == "hello world\nagain"
    # Unbracketed clock times inside a sentence are speech, not markers.
    assert clean_text("we meet at 10:30 today") == "we meet at 10:30 today"
    assert clean_text("") == ""
    assert clean_text("[music]") == ""
