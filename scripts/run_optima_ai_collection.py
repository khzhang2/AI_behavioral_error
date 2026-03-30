from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from optima_common import (
    AI_COLLECTION_DIR,
    CONFIG,
    CHOICE_CODE_TO_NAME,
    CHOICE_LABEL_TO_CODE,
    DATA_DIR,
    EXPERIMENT_DIR,
    INDICATOR_NAMES,
    OUTPUT_DIR,
    archive_experiment_config,
    ensure_dir,
    parse_choice_label,
    parse_indicator_value,
    read_json,
    write_json,
)
from optima_questionnaire_template import (
    build_choice_prompt,
    build_grounding_prompt,
    build_indicator_prompt,
    build_system_prompt,
    indicator_statement,
)

AI_OUTPUT_DIR = AI_COLLECTION_DIR


class ChatBackend:
    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.provider = str(self.config["provider"]).lower()

    def generate(self, messages: list[dict[str, str]], request_state: dict | None = None) -> dict:
        if self.provider == "ollama":
            return self._generate_ollama(messages)
        return self._generate_openai_compatible(messages)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key", "")
        api_key_env = self.config.get("api_key_env", "")
        if not api_key and api_key_env:
            api_key = os.environ.get(str(api_key_env), "")
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

    def _generate_ollama(self, messages: list[dict[str, str]]) -> dict:
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.get("temperature"),
                "top_p": self.config.get("top_p"),
                "top_k": self.config.get("top_k"),
                "seed": self.config.get("seed"),
                "num_predict": self.config.get("num_predict"),
            },
        }
        if "think" in self.config:
            payload["think"] = self.config["think"]
        if self.config.get("format"):
            payload["format"] = self.config["format"]
        response = self._post_json(str(self.config["base_url"]).rstrip("/") + "/api/chat", payload)
        message = response.get("message", {})
        content = str(message.get("content", ""))
        return {
            "raw_text": content,
            "response_text": content.strip(),
            "thinking_text": "",
            "metadata": {
                "done_reason": response.get("done_reason", ""),
                "total_duration": response.get("total_duration", 0),
                "prompt_eval_count": response.get("prompt_eval_count", 0),
                "eval_count": response.get("eval_count", 0),
            },
        }

    def _generate_openai_compatible(self, messages: list[dict[str, str]]) -> dict:
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p"),
            "seed": self.config.get("seed"),
            "max_tokens": self.config.get("num_predict"),
        }
        if self.config.get("format"):
            payload["response_format"] = self.config["format"]
        if self.config.get("extra_body"):
            payload.update(self.config["extra_body"])
        response = self._post_json(str(self.config["base_url"]).rstrip("/") + "/chat/completions", payload)
        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = str(content)
        usage = response.get("usage", {})
        return {
            "raw_text": content,
            "response_text": content.strip(),
            "thinking_text": "",
            "metadata": {
                "done_reason": choice.get("finish_reason", ""),
                "total_duration": 0,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-respondents", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def build_backends() -> tuple[ChatBackend, ChatBackend, ChatBackend]:
    backend_config = dict(CONFIG["llm"])
    grounding_backend = ChatBackend({**backend_config, "num_predict": int(backend_config["grounding_num_predict"])})
    indicator_backend = ChatBackend({**backend_config, "num_predict": int(backend_config["indicator_num_predict"])})
    choice_backend = ChatBackend({**backend_config, "num_predict": int(backend_config["choice_num_predict"])})
    return grounding_backend, indicator_backend, choice_backend


def previous_answer_strings(indicator_rows: list[dict], choice_row: dict | None = None) -> list[str]:
    items = [f"{row['indicator_name']}={row['indicator_value']}" for row in indicator_rows if row.get("indicator_value", -1) > 0]
    if choice_row and choice_row.get("choice_label"):
        items.append(f"choice={choice_row['choice_label']} ({choice_row['chosen_alternative_name']})")
    return items


def initialize_outputs() -> None:
    ensure_dir(AI_OUTPUT_DIR)
    (AI_OUTPUT_DIR / "raw_interactions.jsonl").write_text("", encoding="utf-8")
    write_json(
        AI_OUTPUT_DIR / "respondent_transcripts.json",
        {"experiment_name": CONFIG["experiment_name"], "respondents": {}},
    )
    write_json(
        AI_OUTPUT_DIR / "run_respondents.json",
        {
            "experiment_name": CONFIG["experiment_name"],
            "target_respondents": 0,
            "completed_respondents": 0,
            "updated_at": now_iso(),
        },
    )


def target_respondent_count(n_respondents: int | None) -> int:
    progress_path = AI_OUTPUT_DIR / "run_respondents.json"
    if n_respondents is not None:
        return int(n_respondents)
    if progress_path.exists():
        progress = read_json(progress_path)
        if int(progress.get("target_respondents", 0)) > 0:
            return int(progress["target_respondents"])
    return int(len(pd.read_csv(DATA_DIR / "human_respondent_profiles.csv")))


def update_progress(target_respondents: int, completed_respondents: int) -> None:
    write_json(
        AI_OUTPUT_DIR / "run_respondents.json",
        {
            "experiment_name": CONFIG["experiment_name"],
            "target_respondents": int(target_respondents),
            "completed_respondents": int(completed_respondents),
            "updated_at": now_iso(),
        },
    )


def load_profiles(n_respondents: int | None) -> pd.DataFrame:
    profiles = pd.read_csv(DATA_DIR / "human_respondent_profiles.csv").copy()
    profiles = profiles.rename(columns={"respondent_id": "human_respondent_id"})
    if n_respondents is not None:
        profiles = profiles.head(int(n_respondents)).copy()
    profiles["ai_respondent_id"] = [f"AI{index + 1:04d}" for index in range(len(profiles))]
    return profiles


def rebuild_from_raw(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, set[str]]:
    raw_rows = read_jsonl(AI_OUTPUT_DIR / "raw_interactions.jsonl")
    transcript_map: dict[str, dict] = {}
    indicator_last: dict[tuple[str, str], dict] = {}
    choice_last: dict[str, dict] = {}
    grounding_last: dict[str, dict] = {}

    for row in raw_rows:
        respondent_id = str(row["respondent_id"])
        transcript_map.setdefault(respondent_id, {"turns": []})
        transcript_map[respondent_id]["turns"].append(
            {
                "stage": row.get("stage"),
                "indicator_name": row.get("indicator_name"),
                "prompt": row.get("prompt"),
                "response_text": row.get("response_text"),
                "metadata": row.get("metadata", {}),
            }
        )
        if row.get("stage") == "grounding":
            grounding_last[respondent_id] = row
        if row.get("stage") in {"indicator", "indicator_repair"} and row.get("indicator_name"):
            indicator_last[(respondent_id, str(row["indicator_name"]))] = row
        if row.get("stage") in {"choice", "choice_repair"}:
            choice_last[respondent_id] = row

    profile_lookup = profiles.set_index("ai_respondent_id").to_dict(orient="index")
    indicator_rows: list[dict] = []
    choice_rows: list[dict] = []
    completed_ids: set[str] = set()

    for respondent_id, row in choice_last.items():
        choice_label = parse_choice_label(str(row.get("response_text", "")))
        choice_code = CHOICE_LABEL_TO_CODE.get(choice_label, -1)
        choice_rows.append(
            {
                "respondent_id": respondent_id,
                "choice_label": choice_label,
                "choice_code": choice_code,
                "chosen_alternative_name": CHOICE_CODE_TO_NAME.get(choice_code, ""),
                "is_valid_choice": int(choice_code in {0, 1, 2}),
                "duration_sec": float(row.get("metadata", {}).get("total_duration", 0)) / 1_000_000_000.0,
            }
        )

    for (respondent_id, indicator_name), row in indicator_last.items():
        indicator_value = parse_indicator_value(str(row.get("response_text", "")))
        indicator_rows.append(
            {
                "respondent_id": respondent_id,
                "indicator_name": indicator_name,
                "indicator_value": indicator_value,
                "is_valid_indicator": int(indicator_value in {1, 2, 3, 4, 5, 6}),
                "duration_sec": float(row.get("metadata", {}).get("total_duration", 0)) / 1_000_000_000.0,
            }
        )

    indicators_frame = pd.DataFrame(indicator_rows)
    choices_frame = pd.DataFrame(choice_rows)

    if not indicators_frame.empty and not choices_frame.empty:
        indicator_counts = indicators_frame.groupby("respondent_id")["is_valid_indicator"].sum()
        choice_valid = choices_frame.set_index("respondent_id")["is_valid_choice"]
        for respondent_id in sorted(set(indicator_counts.index).intersection(set(choice_valid.index))):
            if int(indicator_counts.get(respondent_id, 0)) == len(INDICATOR_NAMES) and int(choice_valid.get(respondent_id, 0)) == 1:
                completed_ids.add(str(respondent_id))

    transcripts = {"experiment_name": CONFIG["experiment_name"], "respondents": {}}
    for respondent_id, payload in transcript_map.items():
        persona = dict(profile_lookup.get(respondent_id, {}))
        if persona:
            persona["respondent_id"] = respondent_id
        transcripts["respondents"][respondent_id] = {"persona": persona, "turns": payload["turns"]}
        if respondent_id in grounding_last:
            transcripts["respondents"][respondent_id]["grounding_response"] = grounding_last[respondent_id].get("response_text", "")

    return indicators_frame, choices_frame, transcripts, completed_ids


def write_ai_dataset(profiles: pd.DataFrame, indicators: pd.DataFrame, choices: pd.DataFrame) -> None:
    indicators = indicators.drop_duplicates(subset=["respondent_id", "indicator_name"], keep="last").copy()
    choices = choices.drop_duplicates(subset=["respondent_id"], keep="last").copy()
    indicator_pivot = indicators.pivot(index="respondent_id", columns="indicator_name", values="indicator_value").reset_index()
    base = profiles.rename(columns={"ai_respondent_id": "respondent_id"}).copy()
    base = base.drop(columns=[column for column in INDICATOR_NAMES + ["Choice"] if column in base.columns], errors="ignore")
    merged = base.merge(
        indicator_pivot,
        on="respondent_id",
        how="left",
    )
    merged = merged.merge(choices[["respondent_id", "choice_code", "choice_label", "chosen_alternative_name"]], on="respondent_id", how="left")
    merged["Choice"] = merged["choice_code"]
    merged.to_csv(AI_OUTPUT_DIR / "ai_cleaned_wide.csv", index=False)


def write_state(profiles: pd.DataFrame, indicators_frame: pd.DataFrame, choices_frame: pd.DataFrame, transcripts: dict) -> None:
    if not indicators_frame.empty:
        indicators_frame = indicators_frame.drop_duplicates(subset=["respondent_id", "indicator_name"], keep="last").copy()
    if not choices_frame.empty:
        choices_frame = choices_frame.drop_duplicates(subset=["respondent_id"], keep="last").copy()
    profiles_out = profiles.copy().rename(columns={"ai_respondent_id": "respondent_id"})
    profiles_out.to_csv(AI_OUTPUT_DIR / "persona_samples.csv", index=False)
    indicators_frame.to_csv(AI_OUTPUT_DIR / "parsed_indicators.csv", index=False)
    choices_frame.to_csv(AI_OUTPUT_DIR / "parsed_choice.csv", index=False)
    write_json(AI_OUTPUT_DIR / "respondent_transcripts.json", transcripts)
    write_ai_dataset(profiles, indicators_frame, choices_frame)


def main() -> None:
    args = parse_args()
    archive_experiment_config(EXPERIMENT_DIR)
    if not args.resume:
        initialize_outputs()

    grounding_backend, indicator_backend, choice_backend = build_backends()
    profiles = load_profiles(target_respondent_count(args.n_respondents))
    raw_path = AI_OUTPUT_DIR / "raw_interactions.jsonl"

    indicators_frame, choices_frame, transcripts, completed_ids = rebuild_from_raw(profiles) if args.resume else (pd.DataFrame(), pd.DataFrame(), {"experiment_name": CONFIG["experiment_name"], "respondents": {}}, set())
    parsed_indicators = indicators_frame.to_dict(orient="records") if not indicators_frame.empty else []
    parsed_choices = choices_frame.to_dict(orient="records") if not choices_frame.empty else []
    total_target = len(profiles)
    update_progress(total_target, len(completed_ids))

    for _, profile in profiles.iterrows():
        if profile["ai_respondent_id"] in completed_ids:
            continue
        parsed_indicators = [row for row in parsed_indicators if row.get("respondent_id") != profile["ai_respondent_id"]]
        parsed_choices = [row for row in parsed_choices if row.get("respondent_id") != profile["ai_respondent_id"]]
        transcripts["respondents"].pop(profile["ai_respondent_id"], None)
        persona = profile.to_dict()
        persona["respondent_id"] = persona["ai_respondent_id"]
        messages: list[dict] = [{"role": "system", "content": build_system_prompt(persona)}]
        turns: list[dict] = []

        grounding_prompt = build_grounding_prompt(persona)
        grounding_messages = messages + [{"role": "user", "content": grounding_prompt}]
        grounding_result = grounding_backend.generate(grounding_messages, request_state={"stage": "grounding"})
        messages.append({"role": "user", "content": grounding_prompt})
        messages.append({"role": "assistant", "content": grounding_result["response_text"]})
        turns.append(
            {
                "stage": "grounding",
                "prompt": grounding_prompt,
                "response_text": grounding_result["response_text"],
                "metadata": grounding_result["metadata"],
            }
        )
        append_jsonl(
            raw_path,
            {
                "respondent_id": persona["respondent_id"],
                "stage": "grounding",
                "prompt": grounding_prompt,
                "messages_payload": grounding_messages,
                "message_count": len(grounding_messages),
                "response_text": grounding_result["response_text"],
                "metadata": grounding_result["metadata"],
            },
        )

        respondent_indicator_rows: list[dict] = []
        for question_index, indicator_name in enumerate(INDICATOR_NAMES, start=1):
            previous_answers = previous_answer_strings(respondent_indicator_rows)
            prompt = build_indicator_prompt(
                indicator_name=indicator_name,
                statement_text=indicator_statement(indicator_name),
                question_index=question_index,
                total_questions=7,
                previous_answers=previous_answers,
            )
            indicator_messages = messages + [{"role": "user", "content": prompt}]
            result = indicator_backend.generate(indicator_messages, request_state={"stage": "indicator", "indicator_name": indicator_name})
            indicator_value = parse_indicator_value(result["response_text"])
            if indicator_value == -1:
                repair_prompt = 'Your last reply was invalid. Reply again with JSON only like {"indicator_value":4}.'
                repair_messages = indicator_messages + [{"role": "assistant", "content": result["response_text"]}, {"role": "user", "content": repair_prompt}]
                result = indicator_backend.generate(repair_messages, request_state={"stage": "indicator_repair", "indicator_name": indicator_name})
                indicator_value = parse_indicator_value(result["response_text"])
                append_jsonl(
                    raw_path,
                    {
                        "respondent_id": persona["respondent_id"],
                        "stage": "indicator_repair",
                        "indicator_name": indicator_name,
                        "prompt": repair_prompt,
                        "messages_payload": repair_messages,
                        "message_count": len(repair_messages),
                        "response_text": result["response_text"],
                        "metadata": result["metadata"],
                    },
                )
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": result["response_text"]})
            row = {
                "respondent_id": persona["respondent_id"],
                "indicator_name": indicator_name,
                "indicator_value": indicator_value,
                "is_valid_indicator": int(indicator_value in {1, 2, 3, 4, 5, 6}),
                "duration_sec": float(result["metadata"].get("total_duration", 0)) / 1_000_000_000.0,
            }
            respondent_indicator_rows.append(row)
            turns.append(
                {
                    "stage": "indicator",
                    "indicator_name": indicator_name,
                    "prompt": prompt,
                    "response_text": result["response_text"],
                    "indicator_value": indicator_value,
                    "metadata": result["metadata"],
                }
            )
            append_jsonl(
                raw_path,
                {
                    "respondent_id": persona["respondent_id"],
                    "stage": "indicator",
                    "indicator_name": indicator_name,
                    "prompt": prompt,
                    "previous_answers": previous_answers,
                    "messages_payload": indicator_messages,
                    "message_count": len(indicator_messages),
                    "response_text": result["response_text"],
                    "metadata": result["metadata"],
                },
            )

        previous_answers = previous_answer_strings(respondent_indicator_rows)
        choice_prompt = build_choice_prompt(persona, question_index=7, total_questions=7, previous_answers=previous_answers)
        choice_messages = messages + [{"role": "user", "content": choice_prompt}]
        choice_result = choice_backend.generate(choice_messages, request_state={"stage": "choice"})
        choice_label = parse_choice_label(choice_result["response_text"])
        if choice_label not in CHOICE_LABEL_TO_CODE:
            repair_prompt = 'Your last reply was invalid. Reply again with JSON only like {"choice_label":"A"}.'
            repair_messages = choice_messages + [{"role": "assistant", "content": choice_result["response_text"]}, {"role": "user", "content": repair_prompt}]
            choice_result = choice_backend.generate(repair_messages, request_state={"stage": "choice_repair"})
            choice_label = parse_choice_label(choice_result["response_text"])
            append_jsonl(
                raw_path,
                {
                    "respondent_id": persona["respondent_id"],
                    "stage": "choice_repair",
                    "prompt": repair_prompt,
                    "messages_payload": repair_messages,
                    "message_count": len(repair_messages),
                    "response_text": choice_result["response_text"],
                    "metadata": choice_result["metadata"],
                },
            )
        choice_code = CHOICE_LABEL_TO_CODE.get(choice_label, -1)
        choice_row = {
            "respondent_id": persona["respondent_id"],
            "choice_label": choice_label,
            "choice_code": choice_code,
            "chosen_alternative_name": CHOICE_CODE_TO_NAME.get(choice_code, ""),
            "is_valid_choice": int(choice_code in {0, 1, 2}),
            "duration_sec": float(choice_result["metadata"].get("total_duration", 0)) / 1_000_000_000.0,
        }
        messages.append({"role": "user", "content": choice_prompt})
        messages.append({"role": "assistant", "content": choice_result["response_text"]})
        turns.append(
            {
                "stage": "choice",
                "prompt": choice_prompt,
                "response_text": choice_result["response_text"],
                "choice_label": choice_label,
                "metadata": choice_result["metadata"],
            }
        )
        append_jsonl(
            raw_path,
            {
                "respondent_id": persona["respondent_id"],
                "stage": "choice",
                "prompt": choice_prompt,
                "previous_answers": previous_answers,
                "messages_payload": choice_messages,
                "message_count": len(choice_messages),
                "response_text": choice_result["response_text"],
                "metadata": choice_result["metadata"],
            },
        )

        parsed_indicators.extend(respondent_indicator_rows)
        parsed_choices.append(choice_row)
        transcripts["respondents"][persona["respondent_id"]] = {"persona": persona, "turns": turns}
        indicators_frame = pd.DataFrame(parsed_indicators)
        choices_frame = pd.DataFrame(parsed_choices)
        completed_ids.add(persona["respondent_id"])
        write_state(profiles, indicators_frame, choices_frame, transcripts)
        update_progress(total_target, len(completed_ids))
        print(f"[ai_collection] completed respondent={persona['respondent_id']} valid_indicators={sum(row['is_valid_indicator'] for row in respondent_indicator_rows)}/6 valid_choice={choice_row['is_valid_choice']}")

    indicators_frame = pd.DataFrame(parsed_indicators)
    choices_frame = pd.DataFrame(parsed_choices)
    write_state(profiles, indicators_frame, choices_frame, transcripts)
    update_progress(total_target, len(completed_ids))


if __name__ == "__main__":
    main()
