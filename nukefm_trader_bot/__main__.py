from __future__ import annotations

import argparse
from pathlib import Path

from .bot import build_bot


def main() -> None:
    parser = argparse.ArgumentParser(prog="nukefm-trader-bot")
    parser.add_argument("--config", type=Path, default=Path("bots/trader/config.json"))
    arguments = parser.parse_args()

    bot = build_bot(arguments.config)
    bot.run_once()


if __name__ == "__main__":
    main()
