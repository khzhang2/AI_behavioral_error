from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / ".python_packages" / "cu118",):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import torch

from scripts.legacy_estimate_optima_biogeme_hcm import evaluate_loglikelihood as evaluate_biogeme_loglikelihood, load_dataset as load_hcm_dataset
from scripts.legacy_estimate_optima_torch_hcm import evaluate_loglikelihood as evaluate_torch_loglikelihood
from optima_common import DATA_DIR, INDICATOR_NAMES, OUTPUT_DIR, total_variation_distance, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-biogeme-dir", type=str, required=True)
    parser.add_argument("--human-torch-dir", type=str, required=True)
    parser.add_argument("--ai-biogeme-dir", type=str, required=True)
    parser.add_argument("--ai-torch-dir", type=str, required=True)
    parser.add_argument("--aggregate-dir", type=str, required=True)
    parser.add_argument("--n-draws", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def load_estimates(path: Path, estimate_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    if estimate_column != "estimate" and "estimate" in frame.columns:
        frame = frame.rename(columns={"estimate": estimate_column})
    return frame


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, left_col: str, right_col: str) -> tuple[pd.DataFrame, dict]:
    comparison = left.merge(right, on="parameter_name", how="inner")
    comparison["left_sign"] = np.sign(comparison[left_col])
    comparison["right_sign"] = np.sign(comparison[right_col])
    comparison["sign_match"] = comparison["left_sign"] == comparison["right_sign"]
    comparison["abs_diff"] = (comparison[left_col] - comparison[right_col]).abs()
    summary = {
        "n_compared_parameters": int(len(comparison)),
        "n_sign_matches": int(comparison["sign_match"].sum()),
        "sign_match_rate": float(comparison["sign_match"].mean()),
        "mean_abs_difference": float(comparison["abs_diff"].mean()),
        "max_abs_difference": float(comparison["abs_diff"].max()),
    }
    return comparison, summary


def frame_to_mapping(frame: pd.DataFrame, value_col: str) -> dict[str, float]:
    return {row["parameter_name"]: float(row[value_col]) for _, row in frame.iterrows()}


def compare_choice_distribution(human_frame: pd.DataFrame, ai_frame: pd.DataFrame) -> pd.DataFrame:
    human_share = human_frame["Choice"].value_counts(normalize=True).sort_index()
    ai_share = ai_frame["Choice"].value_counts(normalize=True).sort_index()
    rows = []
    for level in sorted(set(human_share.index).union(set(ai_share.index))):
        rows.append(
            {
                "choice_code": int(level),
                "human_share": float(human_share.get(level, 0.0)),
                "ai_share": float(ai_share.get(level, 0.0)),
                "difference": float(ai_share.get(level, 0.0) - human_share.get(level, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def compare_indicator_distribution(human_frame: pd.DataFrame, ai_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for indicator in INDICATOR_NAMES:
        human_share = human_frame[indicator].value_counts(normalize=True).sort_index()
        ai_share = ai_frame[indicator].value_counts(normalize=True).sort_index()
        rows.append(
            {
                "indicator_name": indicator,
                "human_mean": float(human_frame[indicator].mean()),
                "ai_mean": float(ai_frame[indicator].mean()),
                "mean_difference": float(ai_frame[indicator].mean() - human_frame[indicator].mean()),
                "total_variation_distance": float(total_variation_distance(human_share, ai_share)),
            }
        )
    return pd.DataFrame(rows)


def objective_check(dataset: str, biogeme_estimates: pd.DataFrame, torch_estimates: pd.DataFrame, n_draws: int, max_rows: int | None) -> dict:
    frame = load_hcm_dataset(dataset, max_rows)
    draw_path = DATA_DIR / f"shared_sobol_draws_{int(n_draws)}.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    biogeme_params = frame_to_mapping(biogeme_estimates, "biogeme_estimate")
    torch_params = frame_to_mapping(torch_estimates, "torch_estimate")
    biogeme_at_biogeme = evaluate_biogeme_loglikelihood(frame, biogeme_params, draw_path, n_draws)
    biogeme_at_torch = evaluate_biogeme_loglikelihood(frame, torch_params, draw_path, n_draws)
    torch_at_biogeme = evaluate_torch_loglikelihood(frame, biogeme_params, draw_path, n_draws, device)
    torch_at_torch = evaluate_torch_loglikelihood(frame, torch_params, draw_path, n_draws, device)
    diff_biogeme = None if not (math.isfinite(biogeme_at_biogeme) and math.isfinite(torch_at_biogeme)) else abs(biogeme_at_biogeme - torch_at_biogeme)
    diff_torch = None if not (math.isfinite(biogeme_at_torch) and math.isfinite(torch_at_torch)) else abs(biogeme_at_torch - torch_at_torch)
    return {
        "dataset": dataset,
        "n_draws": int(n_draws),
        "n_rows": int(len(frame)),
        "biogeme_at_biogeme": safe_float(biogeme_at_biogeme),
        "biogeme_at_torch": safe_float(biogeme_at_torch),
        "torch_at_biogeme": safe_float(torch_at_biogeme),
        "torch_at_torch": safe_float(torch_at_torch),
        "same_point_diff_at_biogeme": safe_float(diff_biogeme) if diff_biogeme is not None else None,
        "same_point_diff_at_torch": safe_float(diff_torch) if diff_torch is not None else None,
    }


def main() -> None:
    args = parse_args()
    aggregate_dir = OUTPUT_DIR / args.aggregate_dir
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    human_biogeme32 = load_estimates(OUTPUT_DIR / args.human_biogeme_dir / "biogeme_hcm_estimates.csv", "biogeme_estimate")
    ai_biogeme32 = load_estimates(OUTPUT_DIR / args.ai_biogeme_dir / "biogeme_hcm_estimates.csv", "biogeme_estimate")
    human_torch32 = load_estimates(OUTPUT_DIR / args.human_torch_dir / "torch_hcm_estimates.csv", "torch_estimate")
    ai_torch32 = load_estimates(OUTPUT_DIR / args.ai_torch_dir / "torch_hcm_estimates.csv", "torch_estimate")

    human_torch32_vs_biogeme32, human_summary = compare_frames(human_torch32, human_biogeme32, "torch_estimate", "biogeme_estimate")
    ai_torch32_vs_biogeme32, ai_summary = compare_frames(ai_torch32, ai_biogeme32, "torch_estimate", "biogeme_estimate")
    human_torch32_vs_biogeme32.to_csv(aggregate_dir / "human_torch32_vs_biogeme32_comparison.csv", index=False)
    ai_torch32_vs_biogeme32.to_csv(aggregate_dir / "ai_torch32_vs_biogeme32_comparison.csv", index=False)

    human_frame = load_hcm_dataset("human", args.max_rows)
    ai_frame = load_hcm_dataset("ai", args.max_rows)
    compare_choice_distribution(human_frame, ai_frame).to_csv(aggregate_dir / "human_vs_ai_choice_shares.csv", index=False)
    compare_indicator_distribution(human_frame, ai_frame).to_csv(aggregate_dir / "human_vs_ai_indicator_distributions.csv", index=False)

    objective_checks = {
        "human": objective_check("human", human_biogeme32, human_torch32, args.n_draws, args.max_rows),
        "ai": objective_check("ai", ai_biogeme32, ai_torch32, args.n_draws, args.max_rows),
    }
    summary = {
        "human_torch32_vs_biogeme32": human_summary,
        "ai_torch32_vs_biogeme32": ai_summary,
        "same_point_objective_checks": objective_checks,
    }
    write_json(aggregate_dir / "comparison_summary.json", summary)
    write_json(aggregate_dir / "torch32_same_point_objective_checks.json", objective_checks)


if __name__ == "__main__":
    main()
