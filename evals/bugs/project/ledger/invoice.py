from dataclasses import dataclass, field
from decimal import Decimal

from .money import round_cents


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return self.unit_price * self.quantity


@dataclass
class Invoice:
    number: str
    items: list[LineItem] = field(default_factory=list)
    tax_rate: Decimal = Decimal("0")
    discount_pct: Decimal = Decimal("0")

    def add(self, item: LineItem) -> None:
        if item.quantity <= 0:
            raise ValueError("quantity must be positive")
        self.items.append(item)

    @property
    def subtotal(self) -> Decimal:
        return round_cents(sum((i.amount for i in self.items), Decimal("0")))

    @property
    def discount(self) -> Decimal:
        return round_cents(self.subtotal * self.discount_pct / 100)

    @property
    def tax(self) -> Decimal:
        # tax applies to the discounted subtotal
        return round_cents((self.subtotal - self.discount) * self.tax_rate / 100)

    @property
    def total(self) -> Decimal:
        return self.subtotal - self.discount + self.tax
