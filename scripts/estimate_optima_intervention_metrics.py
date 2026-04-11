from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from optima_common import CONFIG, EXPERIMENT_DIR, ai_collection_dir_for, ensure_dir, llm_models, total_variation_distance, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--max-templates-per-model", type=int, default=None)
    return parser.parse_args()


def load_task_responses(max_templates_per_model: int | None) -> pd.DataFrame:
    frames = []
    for model_config in llm_models():
        path = ai_collection_dir_for(model_config["key"]) / "parsed_task_responses.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["model_key"] = model_config["key"]
        if max_templates_per_model is not None:
            keep_templates = frame["block_template_id"].drop_duplicates().tolist()[: int(max_templates_per_model)]
            frame = frame.loc[frame["block_template_id"].isin(keep_templates)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["model_key", "block_template_id", "run_repeat", "task_index"]).reset_index(drop=True)


def exact_repeat_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = frame.loc[frame["is_valid_task_response"] == 1].groupby(["model_key", "block_template_id", "task_index", "task_role", "pair_id", "manipulation_type"])
    for keys, group in grouped:
        choices = group["choice_code"].astype(int).tolist()
        if len(choices) <= 1:
            flip_rate = np.nan
            entropy = np.nan
        else:
            flips = []
            for left in range(len(choices)):
                for right in range(left + 1, len(choices)):
                    flips.append(int(choices[left] != choices[right]))
            shares = pd.Series(choices).value_counts(normalize=True)
            flip_rate = float(np.mean(flips))
            entropy = float(-(shares * np.log(np.clip(shares, 1e-12, None))).sum())
        rows.append(
            {
                "model_key": keys[0],
                "block_template_id": keys[1],
                "task_index": int(keys[2]),
                "task_role": keys[3],
                "pair_id": keys[4],
                "manipulation_type": keys[5],
                "n_repeats": int(len(group)),
                "exact_repeat_flip_rate": flip_rate,
                "response_entropy": entropy,
            }
        )
    return pd.DataFrame(rows)


def intervention_summary(frame: pd.DataFrame, repeat_frame: pd.DataFrame) -> pd.DataFrame:
    randomness_lookup = repeat_frame.set_index(["model_key", "block_template_id", "task_index"])["exact_repeat_flip_rate"].to_dict()
    rows = []
    valid = frame.loc[frame["is_valid_task_response"] == 1].copy()
    utility_equivalent = set(CONFIG["intervention_tests"]["utility_equivalent_manipulations"])
    pair_groups = valid.loc[valid["manipulation_type"].isin(utility_equivalent)].groupby(["model_key", "block_template_id", "pair_id"])
    for keys, group in pair_groups:
        if group["anchor_task_index"].max() < 0:
            continue
        anchor_index = int(group["anchor_task_index"].max())
        anchor = valid.loc[
            (valid["model_key"] == keys[0])
            & (valid["block_template_id"] == keys[1])
            & (valid["task_index"] == anchor_index)
            & (valid["is_valid_task_response"] == 1)
        ].copy()
        twin = group.copy()
        if anchor.empty or twin.empty:
            continue
        anchor_share = anchor["choice_code"].value_counts(normalize=True)
        twin_share = twin["choice_code"].value_counts(normalize=True)
        tv_gap = total_variation_distance(anchor_share, twin_share)
        randomness_envelope = randomness_lookup.get((keys[0], keys[1], anchor_index), np.nan)
        rows.append(
            {
                "model_key": keys[0],
                "block_template_id": keys[1],
                "pair_id": keys[2],
                "manipulation_type": str(twin["manipulation_type"].iloc[0]),
                "anchor_task_index": anchor_index,
                "intervention_gap_tv": float(tv_gap),
                "randomness_envelope": float(randomness_envelope) if randomness_envelope == randomness_envelope else np.nan,
                "excess_intervention_gap": float(tv_gap - float(CONFIG["intervention_tests"]["repeat_randomness_kappa"]) * randomness_envelope)
                if randomness_envelope == randomness_envelope
                else np.nan,
                "anchor_modal_choice": int(anchor["choice_code"].mode().iloc[0]),
                "twin_modal_choice": int(twin["choice_code"].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def block_diagnostics(frame: pd.DataFrame, repeat_frame: pd.DataFrame, intervention_frame: pd.DataFrame) -> pd.DataFrame:
    repeat_by_block = repeat_frame.groupby(["model_key", "block_template_id"])["exact_repeat_flip_rate"].mean().reset_index(name="exact_repeat_flip_rate_mean")
    repeat_by_block["response_entropy_mean"] = repeat_frame.groupby(["model_key", "block_template_id"])["response_entropy"].mean().values
    intervention_by_block = intervention_frame.groupby(["model_key", "block_template_id"]).agg(
        intervention_gap_tv_mean=("intervention_gap_tv", "mean"),
        excess_intervention_gap_mean=("excess_intervention_gap", "mean"),
    ).reset_index()
    block_rows = frame.groupby(["model_key", "block_template_id", "respondent_id", "run_repeat"]).agg(
        paraphrase_flip_rate=("manipulation_type", lambda _: np.nan),
    ).reset_index()
    block_rows = block_rows.drop(columns=["paraphrase_flip_rate"])
    block_rows = block_rows.merge(repeat_by_block, on=["model_key", "block_template_id"], how="left")
    block_rows = block_rows.merge(intervention_by_block, on=["model_key", "block_template_id"], how="left")
    return block_rows


def bootstrap_h0(intervention_frame: pd.DataFrame) -> dict:
    if intervention_frame.empty:
        return {"n_templates": 0, "bootstrap_repetitions": 0, "mean_excess_gap": np.nan}
    rng = np.random.default_rng(int(CONFIG["master_seed"]) + 404)
    template_ids = intervention_frame[["model_key", "block_template_id"]].drop_duplicates().reset_index(drop=True)
    template_keys = list(template_ids.itertuples(index=False, name=None))
    bootstrap_values = []
    n_boot = int(CONFIG["intervention_tests"]["bootstrap_repetitions"])
    for _ in range(n_boot):
        sampled_keys = [template_keys[index] for index in rng.integers(0, len(template_keys), size=len(template_keys))]
        sampled_rows = []
        for key in sampled_keys:
            sampled_rows.append(intervention_frame.loc[(intervention_frame["model_key"] == key[0]) & (intervention_frame["block_template_id"] == key[1])])
        pooled = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()
        bootstrap_values.append(float(pooled["excess_intervention_gap"].mean()) if not pooled.empty else np.nan)
    bootstrap_series = pd.Series(bootstrap_values).dropna()
    return {
        "n_templates": int(len(template_keys)),
        "bootstrap_repetitions": n_boot,
        "mean_excess_gap": float(intervention_frame["excess_intervention_gap"].mean()),
        "bootstrap_mean_excess_gap": float(bootstrap_series.mean()) if not bootstrap_series.empty else np.nan,
        "bootstrap_ci_lower": float(bootstrap_series.quantile(0.025)) if not bootstrap_series.empty else np.nan,
        "bootstrap_ci_upper": float(bootstrap_series.quantile(0.975)) if not bootstrap_series.empty else np.nan,
        "rejects_pure_randomness_h0": bool(bootstrap_series.quantile(0.025) > 0) if not bootstrap_series.empty else False,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(EXPERIMENT_DIR / "outputs" / args.output_subdir)
    frame = load_task_responses(args.max_templates_per_model)
    if frame.empty:
        raise RuntimeError("No parsed task responses found for intervention diagnostics.")

    repeat_frame = exact_repeat_summary(frame)
    intervention_frame = intervention_summary(frame, repeat_frame)
    block_frame = block_diagnostics(frame, repeat_frame, intervention_frame)

    repeat_frame.to_csv(output_dir / "exact_repeat_randomness.csv", index=False)
    intervention_frame.to_csv(output_dir / "intervention_sensitivity.csv", index=False)
    block_frame.to_csv(output_dir / "block_intervention_diagnostics.csv", index=False)

    summary = {
        "n_rows": int(len(frame)),
        "n_valid_rows": int(frame["is_valid_task_response"].sum()),
        "n_block_templates": int(frame["block_template_id"].nunique()),
        "mean_exact_repeat_flip_rate": float(repeat_frame["exact_repeat_flip_rate"].mean()),
        "mean_response_entropy": float(repeat_frame["response_entropy"].mean()),
        "mean_intervention_gap_tv": float(intervention_frame["intervention_gap_tv"].mean()) if not intervention_frame.empty else np.nan,
        "mean_excess_intervention_gap": float(intervention_frame["excess_intervention_gap"].mean()) if not intervention_frame.empty else np.nan,
        "h0_test": bootstrap_h0(intervention_frame),
    }
    write_json(output_dir / "intervention_metrics_summary.json", summary)
    print(
        f"[estimate_optima_intervention_metrics] templates={summary['n_block_templates']} "
        f"mean_repeat_flip={summary['mean_exact_repeat_flip_rate']:.4f} mean_excess_gap={summary['mean_excess_intervention_gap']:.4f}"
    )


if __name__ == "__main__":
    main()
