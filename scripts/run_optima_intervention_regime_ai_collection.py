from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from optima_common import (
    CONFIG,
    EXPERIMENT_DIR,
    OUTPUT_DIR,
    TASK_ATTRIBUTE_OPTIONS,
    archive_experiment_config,
    configured_indicator_names,
    decode_chat_response,
    ensure_dir,
    experiment_artifact_path,
    llm_config_for,
    parse_choice_label,
    parse_indicator_value,
    parse_task_response,
    raw_output_path,
    resolve_llm_api_key,
    read_json,
    survey_total_tasks,
    write_json,
)
from optima_intervention_regime_questionnaire import (
    build_attitude_prompt,
    build_grounding_prompt,
    build_system_prompt,
    build_task_prompt,
)


CHOICE_NAME_TO_CODE = {"PT": 0, "CAR": 1, "SLOW_MODES": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-repeats", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def append_jsonl_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


class ChatBackend:
    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.provider = str(self.config["provider"]).lower()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = resolve_llm_api_key(self.config)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post_json(self, url: str, payload: dict) -> dict:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        timeout = int(self.config.get("timeout_sec", 240))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Backend request failed: {exc.code} {body}") from exc

    def generate(self, messages: list[dict[str, str]], num_predict: int) -> dict:
        if self.provider == "ollama":
            payload = {
                "model": self.config["model"],
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.get("temperature"),
                    "top_p": self.config.get("top_p"),
                    "top_k": self.config.get("top_k"),
                    "seed": self.config.get("seed"),
                    "num_predict": int(num_predict),
                },
            }
            if "think" in self.config:
                payload["think"] = self.config["think"]
            if self.config.get("format"):
                payload["format"] = self.config["format"]
            response = self._post_json(str(self.config["base_url"]).rstrip("/") + "/api/chat", payload)
            decoded = decode_chat_response(response, self.config, "ollama")
            return {
                "response_text": decoded["response_text"],
                "thinking_text": decoded["thinking_text"],
                "metadata": decoded["metadata"],
            }

        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p"),
            "seed": self.config.get("seed"),
            "max_tokens": int(num_predict),
        }
        reasoning_effort = str(self.config.get("reasoning_effort", "")).strip()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        format_value = self.config.get("format")
        if isinstance(format_value, dict):
            payload["response_format"] = format_value
        elif isinstance(format_value, str) and format_value.strip():
            if format_value.strip().lower() in {"json", "json_object"}:
                payload["response_format"] = {"type": "json_object"}
            else:
                payload["response_format"] = format_value
        if self.config.get("extra_body"):
            payload.update(self.config["extra_body"])
        base_url = str(self.config["base_url"]).rstrip("/")
        if self.provider == "poe" and not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        response = self._post_json(base_url + "/chat/completions", payload)
        decoded = decode_chat_response(response, self.config, self.provider)
        return {
            "response_text": decoded["response_text"],
            "thinking_text": decoded["thinking_text"],
            "metadata": decoded["metadata"],
        }


def initialize_outputs(base_dir: Path, raw_dir: Path, experiment_name: str, target_respondents: int) -> None:
    ensure_dir(base_dir)
    ensure_dir(raw_dir)
    (raw_dir / "raw_interactions.jsonl").write_text("", encoding="utf-8")
    for filename in [
        "persona_samples.csv",
        "parsed_attitudes.csv",
        "parsed_task_responses.csv",
        "ai_panel_long.csv",
        "ai_panel_block.csv",
    ]:
        target = base_dir / filename
        if target.exists():
            target.unlink()
    write_json(raw_dir / "respondent_transcripts.json", {"experiment_name": experiment_name, "respondents": {}})
    write_json(
        raw_dir / "run_respondents.json",
        {
            "experiment_name": experiment_name,
            "target_respondents": int(target_respondents),
            "completed_respondents": 0,
            "updated_at": now_iso(),
        },
    )


def completed_ids(base_dir: Path, total_tasks: int) -> set[str]:
    parsed_path = base_dir / "parsed_task_responses.csv"
    if not parsed_path.exists():
        return set()
    frame = pd.read_csv(parsed_path)
    if frame.empty:
        return set()
    counts = frame.groupby("respondent_id")["task_index"].nunique()
    return {str(index) for index, value in counts.items() if int(value) >= total_tasks}


def respondent_ids_with_any_data(raw_path: Path, attitudes_path: Path, tasks_path: Path) -> set[str]:
    respondent_ids = {str(row.get("respondent_id", "")) for row in read_jsonl(raw_path) if str(row.get("respondent_id", "")).strip()}
    if attitudes_path.exists() and attitudes_path.stat().st_size > 0:
        attitudes = pd.read_csv(attitudes_path)
        if "respondent_id" in attitudes.columns:
            respondent_ids.update(attitudes["respondent_id"].astype(str).tolist())
    if tasks_path.exists() and tasks_path.stat().st_size > 0:
        tasks = pd.read_csv(tasks_path)
        if "respondent_id" in tasks.columns:
            respondent_ids.update(tasks["respondent_id"].astype(str).tolist())
    return respondent_ids


def purge_partial_respondents(raw_path: Path, attitudes_path: Path, tasks_path: Path, respondent_ids: set[str]) -> None:
    if not respondent_ids:
        return

    raw_rows = [row for row in read_jsonl(raw_path) if str(row.get("respondent_id", "")) not in respondent_ids]
    raw_path.write_text("", encoding="utf-8")
    for row in raw_rows:
        append_jsonl(raw_path, row)

    if attitudes_path.exists() and attitudes_path.stat().st_size > 0:
        attitudes = pd.read_csv(attitudes_path)
        if "respondent_id" in attitudes.columns:
            attitudes = attitudes.loc[~attitudes["respondent_id"].astype(str).isin(respondent_ids)].copy()
        attitudes.to_csv(attitudes_path, index=False)

    if tasks_path.exists() and tasks_path.stat().st_size > 0:
        tasks = pd.read_csv(tasks_path)
        if "respondent_id" in tasks.columns:
            tasks = tasks.loc[~tasks["respondent_id"].astype(str).isin(respondent_ids)].copy()
        tasks.to_csv(tasks_path, index=False)

    transcript_path = raw_output_path("respondent_transcripts.json")
    if transcript_path.exists():
        payload = read_json(transcript_path)
        payload.setdefault("respondents", {})
        for respondent_id in respondent_ids:
            payload["respondents"].pop(respondent_id, None)
        write_json(transcript_path, payload)


def previous_answer_strings(attitude_rows: list[dict], task_rows: list[dict]) -> list[str]:
    history = [f"{row['indicator_name']}={row['indicator_value']}" for row in attitude_rows if int(row["indicator_value"]) in {1, 2, 3, 4, 5, 6}]
    for row in task_rows:
        if int(row["is_valid_task_response"]) == 1:
            history.append(f"T{int(row['task_index'])}={row['choice_label']}/{row['chosen_alternative_name']}/conf={int(row['confidence'])}")
    return history


def chosen_alternative_name(task_row: pd.Series, choice_label: str) -> str:
    if choice_label not in {"A", "B", "C"}:
        return ""
    return str(task_row[f"display_{choice_label}_alt"])


def update_progress(base_dir: Path, experiment_name: str, target_respondents: int, completed_respondents: int) -> None:
    write_json(
        raw_output_path("run_respondents.json"),
        {
            "experiment_name": experiment_name,
            "target_respondents": int(target_respondents),
            "completed_respondents": int(completed_respondents),
            "updated_at": now_iso(),
        },
    )


def update_transcripts(base_dir: Path, respondent_id: str, persona: dict, turns: list[dict]) -> None:
    path = raw_output_path("respondent_transcripts.json")
    payload = read_json(path) if path.exists() else {"experiment_name": CONFIG["experiment_name"], "respondents": {}}
    payload.setdefault("respondents", {})
    payload["respondents"][respondent_id] = {"persona": json_safe(persona), "turns": json_safe(turns)}
    write_json(path, payload)


def collect_one_respondent(
    model_key: str,
    llm_config: dict,
    persona: dict,
    respondent_tasks: list[dict],
    indicator_names: list[str],
    total_tasks: int,
) -> dict:
    backend = ChatBackend(llm_config)
    respondent_id = str(persona["respondent_id"])
    system_prompt = build_system_prompt(persona, str(persona["prompt_arm"]), str(persona["prompt_family"]))
    messages = [{"role": "system", "content": system_prompt}]
    transcript_turns: list[dict] = []
    raw_rows: list[dict] = []

    attitude_rows: list[dict] = []
    task_rows: list[dict] = []
    turn_index = 1

    grounding_prompt = build_grounding_prompt(persona)
    messages.append({"role": "user", "content": grounding_prompt})
    grounding_response = backend.generate(messages, int(llm_config["grounding_num_predict"]))
    messages.append({"role": "assistant", "content": grounding_response["response_text"]})
    grounding_payload = {
        "model_key": model_key,
        "respondent_id": respondent_id,
        "block_template_id": persona["block_template_id"],
        "run_repeat": int(persona["run_repeat"]),
        "stage": "grounding",
        "turn_index": turn_index,
        "prompt": grounding_prompt,
        "messages_payload": messages[:-1],
        "response_text": grounding_response["response_text"],
        "metadata": grounding_response["metadata"],
        "prompt_len": len(grounding_prompt),
        "message_count": len(messages) - 1,
    }
    raw_rows.append(grounding_payload)
    transcript_turns.append(grounding_payload)

    for question_offset, indicator_name in enumerate(indicator_names, start=1):
        prompt = build_attitude_prompt(indicator_name, question_offset, len(indicator_names) + total_tasks, previous_answer_strings(attitude_rows, task_rows))
        message_snapshot = list(messages)
        messages.append({"role": "user", "content": prompt})
        response = backend.generate(messages, int(llm_config["attitude_num_predict"]))
        messages.append({"role": "assistant", "content": response["response_text"]})
        indicator_value = parse_indicator_value(response["response_text"])
        turn_index += 1
        row = {
            "model_key": model_key,
            "respondent_id": respondent_id,
            "block_template_id": persona["block_template_id"],
            "run_repeat": int(persona["run_repeat"]),
            "indicator_name": indicator_name,
            "indicator_value": indicator_value,
            "is_valid_indicator": int(indicator_value in {1, 2, 3, 4, 5, 6}),
            "duration_sec": float(response["metadata"].get("total_duration", 0)) / 1_000_000_000.0,
        }
        attitude_rows.append(row)
        interaction_row = {
            "model_key": model_key,
            "respondent_id": respondent_id,
            "block_template_id": persona["block_template_id"],
            "run_repeat": int(persona["run_repeat"]),
            "stage": "attitude",
            "turn_index": turn_index,
            "indicator_name": indicator_name,
            "prompt": prompt,
            "messages_payload": message_snapshot,
            "response_text": response["response_text"],
            "metadata": response["metadata"],
            "prompt_len": len(prompt),
            "message_count": len(message_snapshot),
            "parsed_indicator_value": indicator_value,
        }
        raw_rows.append(interaction_row)
        transcript_turns.append(interaction_row)

    for task_row in respondent_tasks:
        prompt = build_task_prompt(task_row, len(indicator_names) + int(task_row["task_index"]), len(indicator_names) + total_tasks, previous_answer_strings(attitude_rows, task_rows))
        message_snapshot = list(messages)
        messages.append({"role": "user", "content": prompt})
        response = backend.generate(messages, int(llm_config["task_num_predict"]))
        messages.append({"role": "assistant", "content": response["response_text"]})
        parsed = parse_task_response(response["response_text"])
        choice_label = parsed["choice_label"] or parse_choice_label(response["response_text"])
        chosen_name = chosen_alternative_name(task_row, choice_label)
        choice_code = CHOICE_NAME_TO_CODE.get(chosen_name, -1)
        top_attributes = list(parsed["top_attributes"])[:2]
        while len(top_attributes) < 2:
            top_attributes.append("")
        turn_index += 1
        row = {
            "model_key": model_key,
            "respondent_id": respondent_id,
            "block_template_id": task_row["block_template_id"],
            "run_repeat": int(task_row["run_repeat"]),
            "task_index": int(task_row["task_index"]),
            "task_role": task_row["task_role"],
            "pair_id": task_row["pair_id"],
            "manipulation_type": task_row["manipulation_type"],
            "anchor_task_index": int(task_row["anchor_task_index"]),
            "is_utility_equivalent_intervention": int(task_row["is_utility_equivalent_intervention"]),
            "choice_label": choice_label,
            "choice_code": choice_code,
            "chosen_alternative_name": chosen_name,
            "confidence": int(parsed["confidence"]),
            "top_attribute_1": top_attributes[0],
            "top_attribute_2": top_attributes[1],
            "dominated_option_seen": parsed["dominated_option_seen"],
            "semantic_labels": int(task_row["semantic_labels"]),
            "option_order": task_row["option_order"],
            "target_alternative_name": task_row["target_alternative_name"],
            "dominated_alternative_name": task_row["dominated_alternative_name"],
            "is_valid_task_response": int(
                choice_code in {0, 1, 2}
                and 1 <= int(parsed["confidence"]) <= 5
                and top_attributes[0] in TASK_ATTRIBUTE_OPTIONS
                and top_attributes[1] in TASK_ATTRIBUTE_OPTIONS
                and top_attributes[0] != top_attributes[1]
                and parsed["dominated_option_seen"] is not None
            ),
            "duration_sec": float(response["metadata"].get("total_duration", 0)) / 1_000_000_000.0,
        }
        task_rows.append(row)
        interaction_row = {
            "model_key": model_key,
            "respondent_id": respondent_id,
            "block_template_id": task_row["block_template_id"],
            "run_repeat": int(task_row["run_repeat"]),
            "stage": "task",
            "turn_index": turn_index,
            "task_index": int(task_row["task_index"]),
            "task_role": task_row["task_role"],
            "pair_id": task_row["pair_id"],
            "manipulation_type": task_row["manipulation_type"],
            "prompt": prompt,
            "messages_payload": message_snapshot,
            "response_text": response["response_text"],
            "metadata": response["metadata"],
            "prompt_len": len(prompt),
            "message_count": len(message_snapshot),
            "parsed_task_response": row,
        }
        raw_rows.append(interaction_row)
        transcript_turns.append(interaction_row)

    return {
        "respondent_id": respondent_id,
        "persona": persona,
        "attitude_rows": attitude_rows,
        "task_rows": task_rows,
        "raw_rows": raw_rows,
        "turns": transcript_turns,
    }


def persist_respondent_result(
    result: dict,
    attitudes_path: Path,
    tasks_path: Path,
    raw_path: Path,
    completed: set[str],
    total_target: int,
) -> None:
    append_csv(attitudes_path, result["attitude_rows"])
    append_csv(tasks_path, result["task_rows"])
    append_jsonl_rows(raw_path, result["raw_rows"])
    update_transcripts(EXPERIMENT_DIR, result["respondent_id"], result["persona"], result["turns"])
    completed.add(result["respondent_id"])
    update_progress(EXPERIMENT_DIR, CONFIG["experiment_name"], total_target, len(completed))
    valid_attitudes = sum(int(row["is_valid_indicator"]) for row in result["attitude_rows"])
    valid_tasks = sum(int(row["is_valid_task_response"]) for row in result["task_rows"])
    print(
        f"[intervention_regime_ai_collection] model={result['task_rows'][0]['model_key'] if result['task_rows'] else ''} "
        f"respondent={result['respondent_id']} valid_attitudes={valid_attitudes}/{len(result['attitude_rows'])} "
        f"valid_tasks={valid_tasks}/{len(result['task_rows'])}"
    )


def build_ai_panel_long(block_frame: pd.DataFrame, task_frame: pd.DataFrame, response_frame: pd.DataFrame) -> pd.DataFrame:
    response_keep = response_frame.drop(
        columns=[column for column in ["semantic_labels", "option_order", "target_alternative_name", "dominated_alternative_name"] if column in response_frame.columns],
        errors="ignore",
    )
    merged = task_frame.merge(
        response_keep,
        on=[
            "model_key",
            "respondent_id",
            "block_template_id",
            "run_repeat",
            "task_index",
            "task_role",
            "pair_id",
            "manipulation_type",
            "anchor_task_index",
        ],
        how="left",
    )
    merged = merged.merge(
        block_frame[
            [
                "model_key",
                "respondent_id",
                "block_template_id",
                "run_repeat",
                "block_complexity_mean",
                "age_30_less",
                "high_education",
                "ScaledIncome",
            ]
        ],
        on=["model_key", "respondent_id", "block_template_id", "run_repeat"],
        how="left",
    )
    rows: list[dict] = []
    for _, row in merged.iterrows():
        for display_label in ["A", "B", "C"]:
            alternative_name = str(row[f"display_{display_label}_alt"])
            alt_code = CHOICE_NAME_TO_CODE[alternative_name]
            rows.append(
                {
                    "model_key": row["model_key"],
                    "model_is_deepseek": int(str(row["model_key"]) == "deepseek_r1_8b"),
                    "respondent_id": row["respondent_id"],
                    "block_template_id": row["block_template_id"],
                    "run_repeat": int(row["run_repeat"]),
                    "human_respondent_id": row["human_respondent_id"],
                    "normalized_weight": float(row["normalized_weight"]),
                    "prompt_arm": row["prompt_arm"],
                    "semantic_arm": int(row["semantic_arm"]),
                    "prompt_family": row["prompt_family"],
                    "prompt_family_naturalistic": int(row["prompt_family_naturalistic"]),
                    "task_index": int(row["task_index"]),
                    "task_role": row["task_role"],
                    "pair_id": row["pair_id"],
                    "manipulation_type": row["manipulation_type"],
                    "anchor_task_index": int(row["anchor_task_index"]),
                    "block_complexity_mean": float(row["block_complexity_mean"]),
                    "age_30_less": int(row["age_30_less"]),
                    "high_education": int(row["high_education"]),
                    "ScaledIncome": float(row["ScaledIncome"]),
                    "CAR_AVAILABLE": int(row["CAR_AVAILABLE"]),
                    "alternative_name": alternative_name,
                    "alternative_code": alt_code,
                    "display_label": display_label,
                    "display_position": {"A": 1, "B": 2, "C": 3}[display_label],
                    "chosen": int(str(row.get("choice_label", "")) == display_label),
                    "observed_choice_code": int(row.get("choice_code", -1)),
                    "is_valid_task_response": int(row.get("is_valid_task_response", 0)),
                    "alt_available": 1 if alternative_name != "CAR" else int(row["CAR_AVAILABLE"]),
                    "alt_time": float(row["TimePT"] if alternative_name == "PT" else row["TimeCar"] if alternative_name == "CAR" else 0.0),
                    "alt_waiting": float(row["WaitingTimePT"] if alternative_name == "PT" else 0.0),
                    "alt_cost": float(row["MarginalCostPT"] if alternative_name == "PT" else row["CostCarCHF"] if alternative_name == "CAR" else 0.0),
                    "alt_distance": float(row["distance_km"] if alternative_name == "SLOW_MODES" else 0.0),
                    "target_alternative_name": row["target_alternative_name"],
                    "dominated_alternative_name": row["dominated_alternative_name"],
                }
            )
    return pd.DataFrame(rows)


def pairwise_flip_rate(values: list[int]) -> float:
    if len(values) <= 1:
        return np.nan
    flips = []
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            flips.append(int(values[left_index] != values[right_index]))
    return float(np.mean(flips)) if flips else np.nan


def response_entropy(values: list[int]) -> float:
    if not values:
        return np.nan
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log(np.clip(counts, 1e-12, None))).sum())


def build_ai_panel_block(block_frame: pd.DataFrame, response_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    response_lookup = {(str(row["respondent_id"]), int(row["task_index"])): row for _, row in response_frame.iterrows()}
    exact_repeat_source = response_frame.loc[response_frame["is_valid_task_response"] == 1].copy()
    randomness_by_template_task = (
        exact_repeat_source.groupby(["block_template_id", "task_index"])["choice_code"].apply(list).reset_index(name="choices")
    )
    randomness_map = {}
    for _, row in randomness_by_template_task.iterrows():
        randomness_map[(str(row["block_template_id"]), int(row["task_index"]))] = {
            "flip_rate": pairwise_flip_rate(list(row["choices"])),
            "entropy": response_entropy(list(row["choices"])),
        }

    for _, block in block_frame.iterrows():
        respondent_id = str(block["respondent_id"])
        rows = response_frame.loc[response_frame["respondent_id"] == respondent_id].copy().sort_values("task_index").reset_index(drop=True)
        valid_rows = rows.loc[rows["is_valid_task_response"] == 1].copy()
        label_flip_values = []
        order_flip_values = []
        paraphrase_flip_values = []
        monotonicity_values = []
        dominance_values = []

        for _, twin_row in rows.loc[rows["task_role"] == "label_mask_twin"].iterrows():
            anchor_key = (respondent_id, int(twin_row["anchor_task_index"]))
            if anchor_key in response_lookup and int(response_lookup[anchor_key]["is_valid_task_response"]) == 1 and int(twin_row["is_valid_task_response"]) == 1:
                label_flip_values.append(int(response_lookup[anchor_key]["choice_code"] != twin_row["choice_code"]))

        for _, twin_row in rows.loc[rows["task_role"] == "order_twin"].iterrows():
            anchor_key = (respondent_id, int(twin_row["anchor_task_index"]))
            if anchor_key in response_lookup and int(response_lookup[anchor_key]["is_valid_task_response"]) == 1 and int(twin_row["is_valid_task_response"]) == 1:
                order_flip_values.append(int(response_lookup[anchor_key]["choice_code"] != twin_row["choice_code"]))

        for _, twin_row in rows.loc[rows["task_role"] == "paraphrase_twin"].iterrows():
            anchor_key = (respondent_id, int(twin_row["anchor_task_index"]))
            if anchor_key in response_lookup and int(response_lookup[anchor_key]["is_valid_task_response"]) == 1 and int(twin_row["is_valid_task_response"]) == 1:
                paraphrase_flip_values.append(int(response_lookup[anchor_key]["choice_code"] != twin_row["choice_code"]))

        for _, twin_row in rows.loc[rows["task_role"] == "monotonicity"].iterrows():
            anchor_key = (respondent_id, int(twin_row["anchor_task_index"]))
            target_name = str(twin_row["target_alternative_name"])
            target_code = CHOICE_NAME_TO_CODE.get(target_name, -1)
            if anchor_key in response_lookup and int(response_lookup[anchor_key]["is_valid_task_response"]) == 1 and int(twin_row["is_valid_task_response"]) == 1:
                anchor_choice = int(response_lookup[anchor_key]["choice_code"])
                twin_choice = int(twin_row["choice_code"])
                monotonicity_values.append(int(not (anchor_choice != target_code and twin_choice == target_code)))

        for _, dom_row in rows.loc[rows["task_role"] == "dominance"].iterrows():
            dominated_code = CHOICE_NAME_TO_CODE.get(str(dom_row["dominated_alternative_name"]), -1)
            if int(dom_row["is_valid_task_response"]) == 1:
                dominance_values.append(int(int(dom_row["choice_code"]) == dominated_code))

        top_attr_counts = {name: 0 for name in TASK_ATTRIBUTE_OPTIONS}
        top_attr_total = 0
        for _, valid_row in valid_rows.iterrows():
            for column in ["top_attribute_1", "top_attribute_2"]:
                name = str(valid_row.get(column, ""))
                if name in top_attr_counts:
                    top_attr_counts[name] += 1
                    top_attr_total += 1

        task_randomness = []
        for _, valid_row in valid_rows.iterrows():
            metrics = randomness_map.get((str(block["block_template_id"]), int(valid_row["task_index"])))
            if metrics is not None:
                task_randomness.append(metrics["flip_rate"])

        record = block.to_dict()
        record.update(
            {
                "model_is_deepseek": int(str(block["model_key"]) == "deepseek_r1_8b"),
                "n_valid_tasks": int(valid_rows["is_valid_task_response"].sum()) if "is_valid_task_response" in valid_rows.columns else 0,
                "label_flip_rate": float(np.mean(label_flip_values)) if label_flip_values else np.nan,
                "order_flip_rate": float(np.mean(order_flip_values)) if order_flip_values else np.nan,
                "paraphrase_flip_rate": float(np.mean(paraphrase_flip_values)) if paraphrase_flip_values else np.nan,
                "monotonicity_compliance_rate": float(np.mean(monotonicity_values)) if monotonicity_values else np.nan,
                "dominance_violation_rate": float(np.mean(dominance_values)) if dominance_values else np.nan,
                "confidence_mean": float(valid_rows["confidence"].mean()) if not valid_rows.empty else np.nan,
                "exact_repeat_flip_rate_mean": float(np.nanmean(task_randomness)) if task_randomness else np.nan,
            }
        )
        for name in TASK_ATTRIBUTE_OPTIONS:
            record[f"top_attr_share_{name}"] = float(top_attr_counts[name] / top_attr_total) if top_attr_total > 0 else np.nan
        records.append(record)
    return pd.DataFrame(records)


def finalize_outputs(base_dir: Path, block_frame: pd.DataFrame, task_frame: pd.DataFrame) -> None:
    attitudes_path = base_dir / "parsed_attitudes.csv"
    tasks_path = base_dir / "parsed_task_responses.csv"
    tasks = pd.read_csv(tasks_path) if tasks_path.exists() and tasks_path.stat().st_size > 0 else pd.DataFrame()
    block_frame.to_csv(base_dir / "persona_samples.csv", index=False)
    if not tasks.empty:
        panel_long = build_ai_panel_long(block_frame, task_frame, tasks)
        panel_block = build_ai_panel_block(block_frame, tasks)
        panel_long.to_csv(base_dir / "ai_panel_long.csv", index=False)
        panel_block.to_csv(base_dir / "ai_panel_block.csv", index=False)
    else:
        pd.DataFrame().to_csv(base_dir / "ai_panel_long.csv", index=False)
        pd.DataFrame().to_csv(base_dir / "ai_panel_block.csv", index=False)

    progress = read_json(raw_output_path("run_respondents.json")) if raw_output_path("run_respondents.json").exists() else {}
    summary = {
        "experiment_name": CONFIG["experiment_name"],
        "model_key": str(block_frame["model_key"].iloc[0]) if not block_frame.empty else "",
        "target_respondents": int(progress.get("target_respondents", 0)),
        "completed_respondents": int(progress.get("completed_respondents", 0)),
        "valid_attitude_rate": float(pd.read_csv(attitudes_path)["is_valid_indicator"].mean()) if attitudes_path.exists() and attitudes_path.stat().st_size > 0 else None,
        "valid_task_rate": float(tasks["is_valid_task_response"].mean()) if not tasks.empty else None,
    }
    write_json(raw_output_path("ai_collection_summary.json"), summary)


def main() -> None:
    args = parse_args()
    archive_experiment_config(EXPERIMENT_DIR)
    llm_config = llm_config_for(args.model_key)
    base_dir = EXPERIMENT_DIR
    raw_dir = OUTPUT_DIR
    indicator_names = configured_indicator_names()
    total_tasks = int(survey_total_tasks())

    block_frame = pd.read_csv(experiment_artifact_path("block_assignments.csv"))
    task_frame = pd.read_csv(experiment_artifact_path("panel_tasks.csv"))
    if args.max_templates is not None:
        keep_templates = block_frame["block_template_id"].drop_duplicates().tolist()[: int(args.max_templates)]
        block_frame = block_frame.loc[block_frame["block_template_id"].isin(keep_templates)].copy()
        task_frame = task_frame.loc[task_frame["block_template_id"].isin(keep_templates)].copy()
    if args.max_repeats is not None:
        block_frame = block_frame.loc[block_frame["run_repeat"] <= int(args.max_repeats)].copy()
        task_frame = task_frame.loc[task_frame["run_repeat"] <= int(args.max_repeats)].copy()
    block_frame = block_frame.sort_values(["block_template_id", "run_repeat"]).reset_index(drop=True)
    task_frame = task_frame.sort_values(["respondent_id", "task_index"]).reset_index(drop=True)

    if not args.resume or not raw_output_path("run_respondents.json").exists():
        initialize_outputs(base_dir, raw_dir, CONFIG["experiment_name"], len(block_frame))
    completed = completed_ids(base_dir, total_tasks)
    update_progress(base_dir, CONFIG["experiment_name"], len(block_frame), len(completed))

    attitudes_path = base_dir / "parsed_attitudes.csv"
    tasks_path = base_dir / "parsed_task_responses.csv"
    raw_path = raw_output_path("raw_interactions.jsonl")
    if args.resume:
        partial_ids = respondent_ids_with_any_data(raw_path, attitudes_path, tasks_path) - completed
        purge_partial_respondents(raw_path, attitudes_path, tasks_path, partial_ids)
    pending_personas = [
        persona.to_dict()
        for _, persona in block_frame.iterrows()
        if str(persona["respondent_id"]) not in completed
    ]
    task_lookup = {
        str(respondent_id): group.sort_values("task_index").to_dict(orient="records")
        for respondent_id, group in task_frame.groupby("respondent_id")
    }
    max_workers = int(args.max_workers if args.max_workers is not None else CONFIG.get("collection", {}).get("max_workers", 1))

    if max_workers <= 1:
        for persona in pending_personas:
            result = collect_one_respondent(args.model_key, llm_config, persona, task_lookup.get(str(persona["respondent_id"]), []), indicator_names, total_tasks)
            persist_respondent_result(result, attitudes_path, tasks_path, raw_path, completed, len(block_frame))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    collect_one_respondent,
                    args.model_key,
                    llm_config,
                    persona,
                    task_lookup.get(str(persona["respondent_id"]), []),
                    indicator_names,
                    total_tasks,
                )
                for persona in pending_personas
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                persist_respondent_result(result, attitudes_path, tasks_path, raw_path, completed, len(block_frame))

    finalize_outputs(base_dir, block_frame, task_frame)


if __name__ == "__main__":
    main()
