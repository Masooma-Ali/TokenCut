"""Pricing helpers."""

TAX_RATE = 0.17


def apply_discount(amount: float, rate: float) -> float:
    return amount * (1 - rate)


def add_tax(amount: float) -> float:
    return amount * (1 + TAX_RATE)
