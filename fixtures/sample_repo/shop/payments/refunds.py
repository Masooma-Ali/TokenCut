"""Refunds. Circular import with gateway.py (deliberate)."""

from shop.payments import gateway


def log_failure(card_number: str) -> None:
    print(f"Payment failed for card ending {card_number[-4:]}")


def refund(cart, card_number: str) -> float:
    if gateway.validate(card_number):
        return cart.total()
    return 0.0
