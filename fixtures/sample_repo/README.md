# sample_repo — test fixture (NOT real code)

Tiny "shop" project with known structure and deliberate traps. Every member tests against this.

| Trap | Where |
|---|---|
| Empty file | `shop/__init__.py` |
| Inheritance | `shop/models.py` (`PremiumCustomer(Customer)`) |
| Cross-file calls | `shop/cart.py` -> `shop/utils/pricing.py` |
| Circular import | `shop/payments/gateway.py` <-> `shop/payments/refunds.py` |
| Same name in two files | `validate` in `shop/cart.py` and `shop/payments/gateway.py` |
| Syntax error | `shop/utils/broken.py` |
| Non-Python file | this README, `config.yaml` |
| Binary file | `shop/logo.bin` |

Very large files are generated inside the tests (tmp_path) so the repo stays small.
