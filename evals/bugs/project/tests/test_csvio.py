from decimal import Decimal

import pytest

from ledger.csvio import read_items

CSV = """Description,Quantity,Unit_Price
widget,2,"$1,000.00"

gadget,1,3.50
"""


def test_reads_rows_and_skips_blank():
    items = read_items(CSV)
    assert [i.description for i in items] == ["widget", "gadget"]
    assert items[0].unit_price == Decimal("1000.00")


def test_missing_column():
    with pytest.raises(ValueError):
        read_items("description,quantity\nx,1\n")
