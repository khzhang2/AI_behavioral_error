from __future__ import annotations

import argparse
import json

from ai_behavioral_error.experiment import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM travel-behavior SP pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-pipeline", help="Run survey, build long format, estimate MNL, and plot summaries.")
    run_parser.add_argument("--config", required=True, help="Path to experiment config JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-pipeline":
        diagnostics = run_pipeline(args.config)
        print(json.dumps(diagnostics, indent=2))
