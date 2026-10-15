"""Command line for the repo index.

    python -m tokencut.repoindex serve  --repo .  [--embed]    MCP server for the agent
    python -m tokencut.repoindex build  --repo .  [--embed]    index now (and embed chunks)
    python -m tokencut.repoindex map    --repo .  [prefix]
    python -m tokencut.repoindex find   --repo .  "issue text"
    python -m tokencut.repoindex symbol --repo .  QuerySet.only
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from tokencut.repoindex.index import DEFAULT_CACHE_DIR, RepoIndex
from tokencut.repoindex.mcp import serve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tokencut.repoindex")
    parser.add_argument("command", choices=["serve", "build", "map", "find", "symbol"])
    parser.add_argument("arg", nargs="?", default="")
    parser.add_argument("--repo", default=".", help="working tree to index")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--embed", action="store_true",
                        help="use dense embeddings too (slow to compute once, cached)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    log = logging.getLogger("tokencut.repoindex")

    start = time.perf_counter()
    index = RepoIndex.open(args.repo, args.cache_dir)
    log.info("indexed %d files, %d chunks in %.1fs", len(index.files), len(index.by_id),
             time.perf_counter() - start)
    if args.embed:
        from tokencut.repoindex.embed import ChunkEmbedder

        index.embedder = ChunkEmbedder(args.cache_dir / "embed")

    if args.command == "serve":
        index.save()
        threading.Thread(target=index.graph, kwargs={"wait": True}, daemon=True).start()
        if index.embedder is not None:  # vectors fill in while the agent works
            threading.Thread(target=index.embedder.embed_missing, args=(index.chunks(),),
                             daemon=True).start()
        serve(index)
        return
    if args.command == "build":
        index.graph(wait=True)
        if index.embedder is not None:
            index.embedder.embed_missing(index.chunks())
        index.save()
        return
    index.graph(wait=True)
    if args.command == "map":
        print(index.repo_map(args.arg))
    elif args.command == "find":
        print(index.find_code(args.arg))
    else:
        print(index.symbol(args.arg))
    index.save()


if __name__ == "__main__":
    main()
