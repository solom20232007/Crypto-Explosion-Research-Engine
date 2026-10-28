import argparse
from pathlib import Path

from src.collector import save_universe, collect_symbols
from src.analysis import analyze_symbols
from src.modeling import train_model
from src.recommendation_monitor import run_monitor
from src.supabase_store import upload_results
from src.supabase_loader import load_from_supabase
from src.prebreakout_advisor import run_advisor
from src.binance_history_collector import collect as collect_history
from src.history_coverage import audit_history


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Explosion Research Engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "universe",
        help="Save active Binance Spot USDT symbols"
    )

    p_collect = sub.add_parser(
        "collect",
        help="Download historical Spot klines"
    )
    p_collect.add_argument("--symbols", nargs="+", required=True)
    p_collect.add_argument("--interval", default="1h")
    p_collect.add_argument("--start", default="2023-10-01")
    p_collect.add_argument("--end", default="2026-10-27")

    p_import = sub.add_parser(
        "load-supabase",
        help="Read existing candle history from Supabase into local CSV files"
    )
    p_import.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p_import.add_argument("--timeframe", default="1h")

    p_analyze = sub.add_parser(
        "analyze",
        help="Build features and detect historical explosions"
    )
    p_analyze.add_argument("--symbols", nargs="*", default=None)

    sub.add_parser(
        "train",
        help="Train baseline with chronological validation and test"
    )
    sub.add_parser(
        "monitor",
        help="Evaluate saved test predictions and post-signal outcomes"
    )
    sub.add_parser(
        "upload",
        help="Upload research results to Supabase"
    )

    args = parser.parse_args()
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    if args.command == "universe":
        path = save_universe()
        print(f"Saved universe: {path}")
    elif args.command == "collect":
        collect_symbols(args.symbols, args.interval, args.start, args.end)
    elif args.command == "load-supabase":
        load_from_supabase(args.symbols, args.timeframe)
    elif args.command == "analyze":
        analyze_symbols(args.symbols)
    elif args.command == "train":
        train_model()
    elif args.command == "monitor":
        run_monitor()
    elif args.command == "upload":
        upload_results()


if __name__ == "__main__":
    main()
