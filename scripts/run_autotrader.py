from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.autotrader import AutoTrader, load_autotrader_config_from_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HELIOS Polymarket autotrader")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation cycle")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--station", action="append", help="Override station allowlist")
    parser.add_argument("--day", type=int, choices=[0, 1], help="Override target day")
    parser.add_argument("--mode", choices=["paper", "live"], help="Override HELIOS_AUTOTRADE_MODE")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


async def _main() -> int:
    load_dotenv()
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_autotrader_config_from_env()
    if args.station:
        config.station_ids = [station.strip().upper() for station in args.station if station.strip()]
    if args.day is not None:
        config.target_day = int(args.day)
    if args.mode:
        config.mode = args.mode

    trader = AutoTrader(config=config)
    if args.loop and not args.once:
        await trader.run_loop()
        return 0

    results = await trader.run_once()
    for row in results:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
