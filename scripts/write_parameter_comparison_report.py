from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = ROOT_DIR / "experiments" / "Swissmetro"
DEFAULT_OUTPUT_NAME = "parameter_comparison_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--experiment-dirs", nargs="*", default=None)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return parser.parse_args()


def fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def experiment_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    if args.experiment_dirs:
        result = []
        for value in args.experiment_dirs:
            path = Path(value)
            if not path.is_absolute():
                path = args.archive_dir / path
            result.append(path.resolve())
        return result

    if not args.archive_dir.exists():
        return []

    return sorted(
        path.resolve()
        for path in args.archive_dir.iterdir()
        if path.is_dir() and (path / "experiment_config.json").exists()
    )


def load_experiment_config(experiment_dir: Path) -> dict:
    path = experiment_dir / "experiment_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def model_name_from_config(config: dict, experiment_dir: Path) -> str:
    models = config.get("llm_models", [])
    if isinstance(models, list) and models:
        model_name = str(models[0].get("model", "")).strip()
        if model_name:
            return model_name
        model_key = str(models[0].get("key", "")).strip()
        if model_key:
            return model_key
    active_key = str(config.get("active_llm_key", "")).strip()
    if active_key:
        return active_key
    return experiment_dir.name


def short_context_text(config: dict, experiment_dir: Path) -> str:
    models = config.get("llm_models", [])
    if not isinstance(models, list) or not models:
        return f"这个文档由 `{experiment_dir.name}` 下的参数对照表自动生成。"
    model = models[0]
    provider = str(model.get("provider", "")).strip()
    key = str(model.get("key", "")).strip()
    parts = [f"这个文档由 `{experiment_dir.name}` 下的参数对照表自动生成。"]
    if provider:
        parts.append(f"当前后端是 `{provider}`。")
    if key:
        parts.append(f"内部模型键是 `{key}`。")
    return " ".join(part.strip() for part in parts if str(part).strip())


def read_parameter_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    if "gap_ai_minus_human" in frame.columns:
        frame["abs_gap_ai_minus_human"] = frame["gap_ai_minus_human"].abs()
    else:
        frame["abs_gap_ai_minus_human"] = pd.NA
    return frame


def top_gap_table(frame: pd.DataFrame, top_n: int = 6) -> list[str]:
    if frame.empty:
        return [
            "| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |",
            "| --- | ---: | ---: | ---: |",
            "| NA | NA | NA | NA |",
        ]
    ordered = frame.sort_values("abs_gap_ai_minus_human", ascending=False).head(top_n)
    lines = [
        "| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['parameter_name']} | {fmt(row.get('human_estimate'))} | {fmt(row.get('ai_estimate'))} | {fmt(row.get('gap_ai_minus_human'))} |"
        )
    return lines


def full_base_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["当前没有找到 `atasoy_2011_replication/parameter_comparison.csv`。"]
    lines = [
        "| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in frame.sort_values("parameter_name").iterrows():
        lines.append(
            f"| {row['parameter_name']} | {fmt(row.get('human_estimate'))} | {fmt(row.get('ai_estimate'))} | {fmt(row.get('gap_ai_minus_human'))} |"
        )
    return lines


def full_hcm_block_tables(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["当前没有找到 `hcm/parameter_comparison.csv`，或者 HCM 当前不可行。"]

    lines: list[str] = []
    for block_name in ["utility", "attitude", "measurement"]:
        block = frame.loc[frame["block"] == block_name].copy()
        if block.empty:
            continue
        lines.append(f"### {block_name}")
        lines.append("")
        lines.append("| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |")
        lines.append("| --- | ---: | ---: | ---: |")
        for _, row in block.sort_values("parameter_name").iterrows():
            lines.append(
                f"| {row['parameter_name']} | {fmt(row.get('human_estimate'))} | {fmt(row.get('ai_estimate'))} | {fmt(row.get('gap_ai_minus_human'))} |"
            )
        lines.append("")

    if not lines:
        return ["当前 `hcm/parameter_comparison.csv` 为空。"]
    return lines[:-1] if lines and lines[-1] == "" else lines


def strongest_gap_sentence(frame: pd.DataFrame, label: str) -> str:
    if frame.empty or "gap_ai_minus_human" not in frame.columns:
        return f"{label} 当前没有可读的参数差值。"
    strongest = frame.sort_values("abs_gap_ai_minus_human", ascending=False).iloc[0]
    direction = "更高" if float(strongest["gap_ai_minus_human"]) > 0 else "更低"
    gap_value = fmt(abs(float(strongest["gap_ai_minus_human"])))
    return (
        f"{label} 当前差值最大的参数是 `{strongest['parameter_name']}`，"
        f"AI 相对 human {direction} `{gap_value}`。"
    )


def build_report_text(experiment_dir: Path, config: dict, base_frame: pd.DataFrame, hcm_frame: pd.DataFrame) -> str:
    model_name = model_name_from_config(config, experiment_dir)
    lines = [f"# {model_name}", ""]
    lines.append(short_context_text(config, experiment_dir))
    lines.append("")
    lines.append(
        "文档读取 `atasoy_2011_replication/parameter_comparison.csv` 与 `hcm/parameter_comparison.csv`，其中 `gap_ai_minus_human` 定义为 AI 参数减 human 参数。"
    )
    lines.append("")
    lines.append(strongest_gap_sentence(base_frame, "Atasoy base logit"))
    lines.append("")
    lines.append(strongest_gap_sentence(hcm_frame, "Exact HCM"))
    lines.append("")
    lines.append("## Atasoy Base Logit 最大差值")
    lines.append("")
    lines.extend(top_gap_table(base_frame))
    lines.append("")
    lines.append("## Atasoy Base Logit 全部参数")
    lines.append("")
    lines.extend(full_base_table(base_frame))
    lines.append("")
    lines.append("## Exact HCM 最大差值")
    lines.append("")
    lines.extend(top_gap_table(hcm_frame))
    lines.append("")
    lines.append("## Exact HCM 分块参数")
    lines.append("")
    lines.extend(full_hcm_block_tables(hcm_frame))
    lines.append("")
    return "\n".join(lines)


def write_report(experiment_dir: Path, output_name: str) -> Path:
    config = load_experiment_config(experiment_dir)
    base_frame = read_parameter_comparison(experiment_dir / "atasoy_2011_replication" / "parameter_comparison.csv")
    hcm_frame = read_parameter_comparison(experiment_dir / "hcm" / "parameter_comparison.csv")
    output_path = experiment_dir / output_name
    output_path.write_text(build_report_text(experiment_dir, config, base_frame, hcm_frame), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    experiment_dirs = experiment_dirs_from_args(args)
    if not experiment_dirs:
        raise RuntimeError("No experiment directories found.")

    written = []
    for experiment_dir in experiment_dirs:
        written.append(write_report(experiment_dir, args.output_name))

    for path in written:
        print(f"[write_parameter_comparison_report] wrote {path}")


if __name__ == "__main__":
    main()
