from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def parse_amount(text: str) -> Decimal:
    """Parse '$1,234.50', '1234.5' or '(12.00)' (negative) into a Decimal."""
    s = text.strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    if not s:
        raise ValueError("empty amount")
    value = Decimal(s)
    return -value if negative else value


def round_cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def format_amount(value: Decimal) -> str:
    v = round_cents(value)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"
