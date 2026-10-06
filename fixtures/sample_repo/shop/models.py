"""Domain models."""


class Product:
    def __init__(self, sku: str, price: float):
        self.sku = sku
        self.price = price


class Customer:
    def __init__(self, name: str):
        self.name = name

    def discount_rate(self) -> float:
        return 0.0


class PremiumCustomer(Customer):
    def discount_rate(self) -> float:
        return 0.1
