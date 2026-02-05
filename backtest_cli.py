#!/usr/bin/env python3
"""
HELIOS Backtest CLI

Command-line interface for running backtests and calibration.

Usage:
    python backtest_cli.py run --station KLGA --start 2026-01-01 --end 2026-01-15
    python backtest_cli.py calibrate --station KLGA --train-days 30 --val-days 7
    python backtest_cli.py report --result data/backtest_results/result.json
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("backtest_cli")


def parse_date(s: str) -> date:
    """Parse date string (YYYY-MM-DD)."""
    return date.fromisoformat(s)


def cmd_run(args):
    """Run a backtest."""
    from core.backtest import (
        BacktestEngine,
        BacktestMode,
        create_conservative_policy,
        create_aggressive_policy,
    )
    from core.backtest.report import ReportGenerator

    logger.info(f"Running backtest: {args.station} from {args.start} to {args.end}")

    # Select mode
    if args.mode == "signal":
        mode = BacktestMode.SIGNAL_ONLY
    else:
        mode = BacktestMode.EXECUTION_AWARE

    # Select policy
    if args.policy == "conservative":
        policy = create_conservative_policy()
    elif args.policy == "aggressive":
        policy = create_aggressive_policy()
    else:
        policy = create_conservative_policy()

    # Create engine
    engine = BacktestEngine(
        policy=policy,
        mode=mode
    )

    # Progress callback
    def on_progress(current, total, message):
        pct = current / total * 100
        print(f"\r[{pct:5.1f}%] {message}".ljust(60), end="", flush=True)

    engine.set_progress_callback(on_progress)

    # Run backtest
    result = engine.run(
        station_id=args.station,
        start_date=parse_date(args.start),
        end_date=parse_date(args.end),
        interval_seconds=args.interval
    )

    print()  # New line after progress

    # Print summary
    print(result.summary())

    # Save results
    result.save(args.output)

    # Generate report if requested
    if args.report:
        generator = ReportGenerator(args.output)
        if args.report == "html":
            path = generator.generate_html(result)
        else:
            path = generator.generate_markdown(result)
        logger.info(f"Report generated: {path}")


def cmd_calibrate(args):
    """Run calibration loop."""
    from core.backtest import BacktestMode
    from core.backtest.calibration import CalibrationLoop, ParameterSet
    from core.backtest.report import ReportGenerator

    # Calculate date ranges
    if args.end:
        end_date = parse_date(args.end)
    else:
        end_date = date.today() - timedelta(days=1)

    total_days = args.train_days + args.val_days
    start_date = end_date - timedelta(days=total_days)

    train_end = start_date + timedelta(days=args.train_days - 1)
    val_start = train_end + timedelta(days=1)
    val_end = end_date

    logger.info(
        f"Calibration: train {start_date} to {train_end}, "
        f"val {val_start} to {val_end}"
    )

    # Create calibration loop
    mode = BacktestMode.SIGNAL_ONLY if args.mode == "signal" else BacktestMode.EXECUTION_AWARE
    cal = CalibrationLoop(mode=mode)

    # Progress callback
    def on_progress(current, total, message):
        pct = current / total * 100
        print(f"\r[{pct:5.1f}%] {message}".ljust(60), end="", flush=True)

    cal.set_progress_callback(on_progress)

    # Run grid search
    result = cal.grid_search(
        station_id=args.station,
        train_start=start_date,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        max_combinations=args.max_combinations
    )

    print()  # New line after progress

    # Print results
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\nBest Parameters: {result.best_params}")
    print(f"Training Score: {result.best_train_score:.4f}")
    print(f"Validation Score: {result.best_val_score:.4f}")
    print(f"Combinations Tested: {result.combinations_tested}/{result.total_combinations}")

    # Save results
    result.save(args.output)

    # Generate report
    generator = ReportGenerator(args.output)
    path = generator.generate_calibration_report(result)
    logger.info(f"Report generated: {path}")


def cmd_report(args):
    """Generate report from saved results."""
    import json
    from core.backtest.engine import BacktestResult, BacktestMode, DayResult
    from core.backtest.metrics import CalibrationMetrics
    from core.backtest.report import ReportGenerator

    # Load result
    with open(args.result, "r") as f:
        data = json.load(f)

    # Reconstruct BacktestResult (simplified)
    # In production, would deserialize fully
    logger.info(f"Loaded result from {args.result}")

    print("\nBacktest Summary:")
    print(f"  Station: {data['config']['station_id']}")
    print(f"  Period: {data['config']['start_date']} to {data['config']['end_date']}")
    print(f"  Mode: {data['config']['mode']}")

    if data.get('trading_summary'):
        print(f"\nTrading Summary:")
        print(f"  Total PnL: ${data['trading_summary']['total_pnl_net']:.2f}")
        print(f"  Win Rate: {data['trading_summary']['win_rate']:.1%}")
        print(f"  Sharpe: {data['trading_summary']['sharpe_ratio']:.2f}")

    if data.get('aggregated_metrics'):
        metrics = data['aggregated_metrics']
        print(f"\nCalibration Metrics:")
        print(f"  Brier Score: {metrics['calibration']['brier_global']:.4f}")
        print(f"  ECE: {metrics['calibration']['ece']:.4f}")


def cmd_list(args):
    """List available data for backtesting."""
    from core.backtest.dataset import get_dataset_builder

    builder = get_dataset_builder()
    dates = builder.list_available_dates(args.station if args.station != "all" else None)

    print(f"\nAvailable dates ({len(dates)} total):")
    for d in dates[-20:]:  # Last 20
        print(f"  {d}")

    if len(dates) > 20:
        print(f"  ... and {len(dates) - 20} more")


def cmd_validate(args):
    """Validate backtest for leakage."""
    from core.backtest.calibration import NoLeakageValidator
    from core.backtest.dataset import get_dataset_builder

    builder = get_dataset_builder()
    validator = NoLeakageValidator()

    logger.info(f"Validating {args.station} on {args.date}")

    # Build dataset
    dataset = builder.build_dataset(args.station, parse_date(args.date))

    # Validate
    is_valid = validator.validate_dataset(dataset)

    print(f"\nValidation Result: {'PASS' if is_valid else 'FAIL'}")
    print(validator.get_report())


def main():
    parser = argparse.ArgumentParser(
        description="HELIOS Backtest CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a signal-only backtest
  python backtest_cli.py run --station KLGA --start 2026-01-01 --end 2026-01-15

  # Run an execution-aware backtest
  python backtest_cli.py run --station KLGA --start 2026-01-01 --end 2026-01-15 --mode execution

  # Run calibration
  python backtest_cli.py calibrate --station KLGA --train-days 30 --val-days 7

  # List available data
  python backtest_cli.py list --station KLGA

  # Validate for data leakage
  python backtest_cli.py validate --station KLGA --date 2026-01-15
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a backtest")
    run_parser.add_argument("--station", required=True, help="Station ID (e.g., KLGA)")
    run_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    run_parser.add_argument(
        "--mode",
        choices=["signal", "execution"],
        default="signal",
        help="Backtest mode"
    )
    run_parser.add_argument(
        "--policy",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Trading policy"
    )
    run_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Timeline interval in seconds"
    )
    run_parser.add_argument(
        "--output",
        default="data/backtest_results",
        help="Output directory"
    )
    run_parser.add_argument(
        "--report",
        choices=["html", "md"],
        help="Generate report format"
    )
    run_parser.set_defaults(func=cmd_run)

    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Run calibration loop")
    cal_parser.add_argument("--station", required=True, help="Station ID")
    cal_parser.add_argument("--end", help="End date (default: yesterday)")
    cal_parser.add_argument("--train-days", type=int, default=30, help="Training period days")
    cal_parser.add_argument("--val-days", type=int, default=7, help="Validation period days")
    cal_parser.add_argument(
        "--mode",
        choices=["signal", "execution"],
        default="signal",
        help="Backtest mode"
    )
    cal_parser.add_argument(
        "--max-combinations",
        type=int,
        default=100,
        help="Maximum parameter combinations"
    )
    cal_parser.add_argument(
        "--output",
        default="data/calibration_results",
        help="Output directory"
    )
    cal_parser.set_defaults(func=cmd_calibrate)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument("--result", required=True, help="Path to result JSON")
    report_parser.set_defaults(func=cmd_report)

    # List command
    list_parser = subparsers.add_parser("list", help="List available data")
    list_parser.add_argument("--station", default="all", help="Station ID or 'all'")
    list_parser.set_defaults(func=cmd_list)

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate for data leakage")
    val_parser.add_argument("--station", required=True, help="Station ID")
    val_parser.add_argument("--date", required=True, help="Date to validate (YYYY-MM-DD)")
    val_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
