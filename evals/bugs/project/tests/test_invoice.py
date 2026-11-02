from decimal import Decimal

import pytest

from ledger.invoice import Invoice, LineItem


def make():
    inv = Invoice("INV-1", tax_rate=Decimal("10"), discount_pct=Decimal("20"))
    inv.add(LineItem("widget", 2, Decimal("50.00")))
    inv.add(LineItem("gadget", 1, Decimal("25.005")))
    return inv


def test_subtotal_rounds():
    assert make().subtotal == Decimal("125.01")


def test_discount_before_tax():
    inv = make()
    assert inv.discount == Decimal("25.00")
    assert inv.tax == Decimal("10.00")
    assert inv.total == Decimal("110.01")


def test_invoices_do_not_share_items():
    a, b = Invoice("A"), Invoice("B")
    a.add(LineItem("x", 1, Decimal("1")))
    assert b.items == []


def test_rejects_zero_quantity():
    with pytest.raises(ValueError):
        Invoice("C").add(LineItem("x", 0, Decimal("1")))
