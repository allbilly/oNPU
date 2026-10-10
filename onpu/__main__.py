from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description="oNPU · local model serving for Rockchip")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", type=Path, default=Path(os.environ.get("ONPU_BACKEND", "~/tinygrad")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ONPU_DATA_DIR", ".data")))
    args = parser.parse_args()
    app = create_app(args.backend, args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_graceful_shutdown=15)


if __name__ == "__main__":
    main()
