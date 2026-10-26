#!/usr/bin/env python3
"""Browse declaration-proof-proof triples from a MathlibPR pairs file.

python3 tools/triplet_view/serve.py
python3 tools/triplet_view/serve.py --pairs data/mathlibpr/pairs.jsonl
"""

from __future__ import annotations

import argparse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE.parents[1] / "data" / "mathlibpr" / "pairs.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--port", type=int, default=8737)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    pairs = args.pairs.expanduser().resolve()
    if not pairs.is_file():
        raise SystemExit(f"Pairs file not found: {pairs}")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(HERE), **kwargs)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/pairs.jsonl":
                super().do_GET()
                return
            data = pairs.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    server = None
    port = args.port
    last_error: OSError | None = None
    for candidate in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise SystemExit(f"Could not bind a port starting at {args.port}: {last_error}")

    url = f"http://127.0.0.1:{port}/"
    print(f"Serving {pairs}", flush=True)
    print(url, flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
