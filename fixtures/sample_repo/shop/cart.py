"""Shopping cart."""

from shop.models import Customer, Product
from shop.utils.pricing import add_tax, apply_discount


def validate(product: Product) -> bool:
    return product.price > 0


class Cart:
    def __init__(self, customer: Customer):
        self.customer = customer
        self.items: list[Product] = []

    def add_item(self, product: Product) -> None:
        if validate(product):
            self.items.append(product)

    def subtotal(self) -> float:
        return sum(p.price for p in self.items)

    def total(self) -> float:
        discounted = apply_discount(self.subtotal(), self.customer.discount_rate())
        return add_tax(discounted)
