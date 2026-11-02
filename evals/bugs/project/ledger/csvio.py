import csv
import io
from decimal import Decimal

from .invoice import LineItem
from .money import parse_amount

REQUIRED = ("description", "quantity", "unit_price")


def read_items(text: str) -> list[LineItem]:
    """Read line items from CSV. Header names are case-insensitive; blank rows are skipped."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    if reader.fieldnames is None:
        return []
    reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]
    missing = [c for c in REQUIRED if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    items = []
    for row in reader:
        if not any((v or "").strip() for v in row.values()):
            continue
        items.append(LineItem(row["description"].strip(), int(row["quantity"]),
                              Decimal(parse_amount(row["unit_price"]))))
    return items
