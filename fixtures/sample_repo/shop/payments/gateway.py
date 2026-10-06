"""Payment gateway. Circular import with refunds.py (deliberate)."""

from shop.cart import Cart
from shop.payments import refunds


def validate(card_number: str) -> bool:
    return len(card_number) == 16 and card_number.isdigit()


def process_payment(cart: Cart, card_number: str) -> float:
    if not validate(card_number):
        refunds.log_failure(card_number)
        raise ValueError("Invalid card")
    return cart.total()
