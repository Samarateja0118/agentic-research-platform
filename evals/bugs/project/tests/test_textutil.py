from ledger.textutil import slugify, truncate


def test_slugify():
    assert slugify("  Hello, World! 2026 ") == "hello-world-2026"


def test_truncate_short_unchanged():
    assert truncate("abc", 5) == "abc"


def test_truncate_exact_width():
    out = truncate("abcdefghij", 8)
    assert out == "abcde..." and len(out) == 8


def test_truncate_tiny_width():
    assert truncate("abcdef", 2) == "ab"
