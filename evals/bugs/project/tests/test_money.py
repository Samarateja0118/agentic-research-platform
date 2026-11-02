from decimal import Decimal

import pytest

from ledger.money import format_amount, parse_amount, round_cents


def test_parse_simple():
    assert parse_amount("12.50") == Decimal("12.50")


def test_parse_thousands_and_symbol():
    assert parse_amount("$1,234.50") == Decimal("1234.50")


def test_parse_negative_parentheses():
    assert parse_amount("(12.00)") == Decimal("-12.00")


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_amount("  $ ")


def test_round_half_up():
    assert round_cents(Decimal("2.345")) == Decimal("2.35")
    assert round_cents(Decimal("2.335")) == Decimal("2.34")


def test_format():
    assert format_amount(Decimal("1234.5")) == "$1,234.50"
    assert format_amount(Decimal("-3")) == "-$3.00"
