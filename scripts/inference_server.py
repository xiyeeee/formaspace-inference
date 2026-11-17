#!/usr/bin/env python3
"""One process, one resident model worker. Never combine with Gradio on the same GPU."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from inference_service.app import create_app
from inference_service.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    settings = Settings.from_env()  # Fail explicitly before server startup if the token is absent.
    settings.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(settings.root / "service.log"), logging.StreamHandler()])
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1, reload=False,
                ws="websockets", ws_max_size=4096, ws_max_queue=8)


if __name__ == "__main__":
    main()
