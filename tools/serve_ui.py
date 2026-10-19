"""Start the local competition demonstration web UI."""

import argparse

from coop.demo_support import EMBEDDING, QWEN
from coop.web_server import serve, start_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("fixture", "qwen"), default="fixture")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--model-path", default=QWEN)
    parser.add_argument("--embedding-path", default=EMBEDDING)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    start_engine(args.backend, args.device, args.embedding_device,
                 model_path=args.model_path, embedding_path=args.embedding_path)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
