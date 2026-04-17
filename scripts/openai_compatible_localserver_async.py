from __future__ import annotations

import asyncio
from typing import Any, Callable

from openai import AsyncOpenAI

from optima_common import (
    TASK_ATTRIBUTE_OPTIONS,
    VALID_INDICATOR_VALUES,
    decode_chat_response,
    parse_choice_label,
    parse_indicator_value,
    parse_task_response,
    resolve_llm_api_key,
)
from optima_intervention_regime_questionnaire import (
    build_attitude_prompt,
    build_grounding_prompt,
    build_system_prompt,
    build_task_prompt,
)


CHOICE_NAME_TO_CODE = {"PT": 0, "CAR": 1, "SLOW_MODES": 2}
LOCALSERVER_BASE_URL_PREFIXES = ("http://10.64.89.161:8000/v1",)
LOCALSERVER_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "grounding_min_tokens": 160,
        "attitude_min_tokens": 128,
        "task_min_tokens": 512,
        "thinking_text_path": "choices.0.message.reasoning",
        "omit_max_tokens": True,
        "max_workers_cap": 1,
    }
}
LOCALSERVER_DEFAULT_PROFILE: dict[str, Any] = {
    "grounding_min_tokens": 0,
    "attitude_min_tokens": 0,
    "task_min_tokens": 0,
    "thinking_text_path": "choices.0.message.reasoning",
    "omit_max_tokens": False,
    "max_workers_cap": 0,
}


def uses_openai_compatible_localserver_async(config: dict[str, Any]) -> bool:
    provider = str(config.get("provider") or "").strip().lower()
    if provider not in {"openai", "openai_compatible"}:
        return False
    base_url = str(config.get("base_url") or "").strip().rstrip("/")
    return any(base_url.startswith(prefix.rstrip("/")) for prefix in LOCALSERVER_BASE_URL_PREFIXES)


def localserver_model_profile(model_name: str) -> dict[str, Any]:
    normalized = str(model_name or "").strip()
    profile = dict(LOCALSERVER_DEFAULT_PROFILE)
    if normalized in LOCALSERVER_MODEL_PROFILES:
        profile.update(LOCALSERVER_MODEL_PROFILES[normalized])
    return profile


def localserver_max_workers_cap(config: dict[str, Any]) -> int:
    if not uses_openai_compatible_localserver_async(config):
        return 0
    profile = localserver_model_profile(str(config.get("model") or ""))
    return int(profile.get("max_workers_cap") or 0)


class OpenAICompatibleAsyncBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.profile = localserver_model_profile(str(self.config.get("model") or ""))
        api_key = resolve_llm_api_key(self.config) or "EMPTY"
        timeout_value = self.config.get("timeout_sec")
        timeout = int(timeout_value) if timeout_value not in (None, "") else 240
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=str(self.config.get("base_url") or "").strip(),
            timeout=timeout,
        )

    def should_omit_max_tokens(self) -> bool:
        return bool(self.profile.get("omit_max_tokens"))

    def effective_num_predict(self, stage: str, num_predict: int) -> int:
        requested = int(num_predict)
        minimum = 0
        if stage == "grounding":
            minimum = int(self.profile.get("grounding_min_tokens") or 0)
        elif stage == "attitude":
            minimum = int(self.profile.get("attitude_min_tokens") or 0)
        elif stage == "task":
            minimum = int(self.profile.get("task_min_tokens") or 0)
        return max(requested, minimum)

    def decoder_config(self) -> dict[str, Any]:
        config_for_decode = dict(self.config)
        decoder = dict(config_for_decode.get("response_decoder") or {})
        thinking_text_path = str(self.profile.get("thinking_text_path") or "").strip()
        if thinking_text_path and not str(decoder.get("thinking_text_path") or "").strip():
            decoder["thinking_text_path"] = thinking_text_path
        config_for_decode["response_decoder"] = decoder
        return config_for_decode

    async def generate(self, messages: list[dict[str, str]], num_predict: int, stage: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p"),
            "seed": self.config.get("seed"),
        }
        effective_num_predict: int | None = None
        if not self.should_omit_max_tokens():
            effective_num_predict = self.effective_num_predict(stage, num_predict)
            payload["max_tokens"] = effective_num_predict
        reasoning_effort = str(self.config.get("reasoning_effort") or "").strip()
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

        extra_body = dict(self.config.get("extra_body") or {})
        response = await self.client.chat.completions.create(
            **payload,
            extra_body=extra_body or None,
        )
        decoded = decode_chat_response(response.model_dump(), self.decoder_config())
        decoded["metadata"]["requested_max_tokens"] = int(num_predict)
        decoded["metadata"]["effective_max_tokens"] = effective_num_predict
        decoded["metadata"]["max_tokens_omitted"] = int(self.should_omit_max_tokens())
        return {
            "response_text": decoded["response_text"],
            "thinking_text": decoded["thinking_text"],
            "metadata": decoded["metadata"],
        }


def previous_answer_strings(attitude_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> list[str]:
    history = [
        f"{row['indicator_name']}={row['indicator_value']}"
        for row in attitude_rows
        if int(row["indicator_value"]) in VALID_INDICATOR_VALUES
    ]
    for row in task_rows:
        if int(row["is_valid_task_response"]) == 1:
            history.append(f"T{int(row['task_index'])}={row['choice_label']}/{row['chosen_alternative_name']}/conf={int(row['confidence'])}")
    return history


def chosen_alternative_name(task_row: dict[str, Any], choice_label: str) -> str:
    if choice_label not in {"A", "B", "C"}:
        return ""
    return str(task_row[f"display_{choice_label}_alt"])


async def collect_one_respondent_async(
    backend: OpenAICompatibleAsyncBackend,
    model_key: str,
    persona: dict[str, Any],
    respondent_tasks: list[dict[str, Any]],
    indicator_names: list[str],
    total_tasks: int,
) -> dict[str, Any]:
    respondent_id = str(persona["respondent_id"])
    system_prompt = build_system_prompt(persona, str(persona["prompt_arm"]), str(persona["prompt_family"]))
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    transcript_turns: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    attitude_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    turn_index = 1

    grounding_prompt = build_grounding_prompt(persona)
    messages.append({"role": "user", "content": grounding_prompt})
    grounding_response = await backend.generate(messages, int(backend.config["grounding_num_predict"]), "grounding")
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
        prompt = build_attitude_prompt(
            indicator_name,
            question_offset,
            len(indicator_names) + total_tasks,
            previous_answer_strings(attitude_rows, task_rows),
        )
        message_snapshot = list(messages)
        messages.append({"role": "user", "content": prompt})
        response = await backend.generate(messages, int(backend.config["attitude_num_predict"]), "attitude")
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
            "is_valid_indicator": int(indicator_value in VALID_INDICATOR_VALUES),
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
        prompt = build_task_prompt(
            task_row,
            len(indicator_names) + int(task_row["task_index"]),
            len(indicator_names) + total_tasks,
            previous_answer_strings(attitude_rows, task_rows),
        )
        message_snapshot = list(messages)
        messages.append({"role": "user", "content": prompt})
        response = await backend.generate(messages, int(backend.config["task_num_predict"]), "task")
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


async def collect_respondents_async(
    model_key: str,
    llm_config: dict[str, Any],
    pending_personas: list[dict[str, Any]],
    task_lookup: dict[str, list[dict[str, Any]]],
    indicator_names: list[str],
    total_tasks: int,
    max_workers: int,
    persist_result: Callable[[dict[str, Any]], None],
) -> None:
    backend = OpenAICompatibleAsyncBackend(llm_config)
    semaphore = asyncio.Semaphore(max_workers)

    async def run_one(persona: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await collect_one_respondent_async(
                backend,
                model_key,
                persona,
                task_lookup.get(str(persona["respondent_id"]), []),
                indicator_names,
                total_tasks,
            )

    tasks = [asyncio.create_task(run_one(persona)) for persona in pending_personas]
    for task in asyncio.as_completed(tasks):
        persist_result(await task)
