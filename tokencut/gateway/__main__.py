"""Run the gateway:  python -m tokencut.gateway   (listens on 127.0.0.1:8787 by default)."""

import os

import uvicorn

from tokencut.gateway.app import create_app

if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=os.environ.get("TOKENCUT_HOST", "127.0.0.1"),  # local only unless asked otherwise
        port=int(os.environ.get("TOKENCUT_PORT", "8787")),
    )
