from __future__ import annotations

import os
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "experiment_config.json"
MODEL_BEHAVIOR_REGISTRY_PATH = ROOT_DIR / "model_behavior_registry.json"
LLM_OPTIONAL_STRING_FIELDS = [
    "base_url",
    "credentials_file",
    "credentials_key",
    "api_key",
    "api_key_env",
    "reasoning_effort",
    "thinking_mode",
]
LLM_OPTIONAL_NULL_FIELDS = ["top_k", "timeout_sec"]
LLM_OPTIONAL_OBJECT_FIELDS = ["extra_body", "response_decoder"]


def _string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, override_value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(merged[key], override_value)
        else:
            merged[key] = override_value
    return merged


def _set_nested_value(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    if not dotted_path:
        return
    keys = [str(part).strip() for part in str(dotted_path).split(".") if str(part).strip()]
    if not keys:
        return
    cursor = payload
    for key in keys[:-1]:
        current = cursor.get(key)
        if not isinstance(current, dict):
            current = {}
            cursor[key] = current
        cursor = current
    cursor[keys[-1]] = value


def _thinking_mode_state(value: Any) -> str:
    normalized = _string_or_empty(value).strip().lower()
    if normalized in {"off", "false", "disabled", "disable", "non_thinking", "none", "no"}:
        return "off"
    if normalized in {"on", "true", "enabled", "enable", "thinking", "yes"}:
        return "on"
    return ""


def _read_model_behavior_registry() -> list[dict[str, Any]]:
    if not MODEL_BEHAVIOR_REGISTRY_PATH.exists():
        return []
    try:
        payload = _read_json_file(MODEL_BEHAVIOR_REGISTRY_PATH)
    except Exception:
        return []
    profiles = payload.get("profiles", [])
    return profiles if isinstance(profiles, list) else []


def model_behavior_profile(config: dict[str, Any]) -> dict[str, Any]:
    model_name = _string_or_empty(config.get("model", "")).strip()
    provider_name = _string_or_empty(config.get("provider", "")).strip().lower()
    for profile in _read_model_behavior_registry():
        if not isinstance(profile, dict):
            continue
        if _string_or_empty(profile.get("model", "")).strip() != model_name:
            continue
        profile_provider = _string_or_empty(profile.get("provider", "")).strip().lower()
        if profile_provider and profile_provider != provider_name:
            continue
        return dict(profile)
    return {}


def apply_model_behavior_profile(config: dict[str, Any]) -> dict[str, Any]:
    merged = normalize_llm_config_shape(config)
    profile = model_behavior_profile(merged)
    if not profile:
        return merged

    decoder_defaults = profile.get("response_decoder_defaults", {})
    if isinstance(decoder_defaults, dict) and decoder_defaults:
        decoder = dict(merged.get("response_decoder") or {})
        for key, value in decoder_defaults.items():
            if key not in decoder or decoder.get(key) in {None, ""}:
                decoder[str(key)] = value
        merged["response_decoder"] = decoder

    thinking_state = _thinking_mode_state(merged.get("thinking_mode", ""))
    thinking_control = profile.get("thinking_control", {})
    if thinking_state and isinstance(thinking_control, dict):
        path = _string_or_empty(thinking_control.get("path", "")).strip()
        mapped_value = thinking_control.get(thinking_state)
        if path != "reasoning_effort":
            merged["reasoning_effort"] = ""
        if path:
            if path == "reasoning_effort":
                merged["reasoning_effort"] = _string_or_empty(mapped_value).strip()
            else:
                extra_body = dict(merged.get("extra_body") or {})
                target_path = path
                if target_path == "extra_body":
                    extra_body = mapped_value if isinstance(mapped_value, dict) else {}
                    merged["extra_body"] = extra_body
                    return normalize_llm_config_shape(merged)
                if target_path.startswith("extra_body."):
                    target_path = target_path[len("extra_body.") :]
                _set_nested_value(extra_body, target_path, mapped_value)
                merged["extra_body"] = extra_body

    return normalize_llm_config_shape(merged)


def _load_experiment_config() -> dict[str, Any]:
    raw = _read_json_file(CONFIG_PATH)
    if not isinstance(raw, dict):
        raise ValueError("experiment_config.json must be a JSON object.")

    base_file = _string_or_empty(raw.get("config_base_file", "")).strip()
    overrides = raw.get("config_overrides")

    if not base_file:
        return raw

    if not isinstance(overrides, dict):
        overrides = {}

    base = _read_json_file(_config_path(base_file))
    if not isinstance(base, dict):
        raise ValueError(f"config_base_file '{base_file}' must point to a JSON object.")

    return _deep_merge(base, overrides)


CONFIG = _load_experiment_config()
SOURCE_DATA_DIR = ROOT_DIR / CONFIG["paths"]["source_data_dir"]
DATA_DIR = SOURCE_DATA_DIR
ARCHIVE_PARENT_DIR = ROOT_DIR / CONFIG["paths"]["archive_dir"]
EXPERIMENT_DIR = ARCHIVE_PARENT_DIR / CONFIG["experiment_name"]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
AI_COLLECTION_DIR = EXPERIMENT_DIR
TIME_SCALE = 200.0
WAIT_SCALE = 60.0
COST_SCALE = 10.0
DISTANCE_SCALE = 5.0
ATASOY_HUMAN_REPLICATION_DIR = DATA_DIR / "atasoy_2011_replication"
ATASOY_BASE_LOGIT_DIR = ATASOY_HUMAN_REPLICATION_DIR / "base_logit"
ATASOY_HCM_DIR = ATASOY_HUMAN_REPLICATION_DIR / "hcm"
SOURCE_OBSERVATION_COLUMN = "source_observation_id"
VALID_INDICATOR_VALUES = (1, 2, 3, 4, 5)

INDICATOR_NAMES = ["Envir01", "Envir02", "Envir05", "Envir06", "Mobil10", "Mobil11", "Mobil16"]
INDICATOR_TEXT = {
    "Envir01": "Fuel prices should be increased to reduce congestion and air pollution.",
    "Envir02": "More public transportation is needed, even if taxes are set to pay the additional costs.",
    "Envir05": "I am concerned about global warming.",
    "Envir06": "Actions and decision making are needed to limit greenhouse gas emissions.",
    "Mobil10": "It is difficult to take the public transport when I travel with my children.",
    "Mobil11": "It is difficult to take the public transport when I carry bags or luggage.",
    "Mobil16": "I do not like changing the mean of transport when I am traveling.",
}
CHOICE_LABEL_TO_CODE = {"A": 0, "B": 1, "C": 2}
CHOICE_CODE_TO_NAME = {0: "PT", 1: "CAR", 2: "SLOW_MODES"}
CHOICE_LABEL_TO_NAME = {"A": "PT", "B": "CAR", "C": "SLOW_MODES"}
TASK_ATTRIBUTE_OPTIONS = ["travel_time", "waiting_time", "cost", "distance", "availability", "mode_label"]


def pt_non_wait_time(total_time: Any, waiting_time: Any):
    value = total_time - waiting_time
    if hasattr(value, "clip"):
        return value.clip(lower=0.0)
    return max(float(value), 0.0)


def ensure_pt_non_wait_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "TimePT" not in work.columns or "WaitingTimePT" not in work.columns:
        return work
    if "TimePT_non_wait" not in work.columns:
        work["TimePT_non_wait"] = pt_non_wait_time(work["TimePT"], work["WaitingTimePT"])
    if "TimePT_non_wait_scaled" not in work.columns:
        work["TimePT_non_wait_scaled"] = work["TimePT_non_wait"] / TIME_SCALE
    return work


def credentials_file_path(credentials_file: str) -> Path:
    path = Path(str(credentials_file))
    return path if path.is_absolute() else ROOT_DIR / path


def normalize_llm_config_shape(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    for field_name in LLM_OPTIONAL_STRING_FIELDS:
        normalized[field_name] = _string_or_empty(normalized.get(field_name, "")).strip()
    format_value = normalized.get("format", "")
    normalized["format"] = "" if format_value is None else format_value
    for field_name in LLM_OPTIONAL_NULL_FIELDS:
        normalized.setdefault(field_name, None)
    for field_name in LLM_OPTIONAL_OBJECT_FIELDS:
        field_value = normalized.get(field_name)
        normalized[field_name] = field_value if isinstance(field_value, dict) else {}
    return normalized


def load_credentials_payload(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_llm_config_shape(config)
    credentials_file = normalized["credentials_file"] or _string_or_empty(config.get("api_key_file", "")).strip()
    credentials_key = normalized["credentials_key"]
    if not credentials_file:
        return {}

    path = credentials_file_path(credentials_file)
    if not path.exists():
        return {}

    try:
        credentials = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if credentials_key:
        nested = credentials.get(credentials_key, {})
        credentials = nested if isinstance(nested, dict) else {}

    return credentials if isinstance(credentials, dict) else {}


def apply_llm_credentials(config: dict[str, Any]) -> dict[str, Any]:
    merged = normalize_llm_config_shape(config)
    credentials = load_credentials_payload(merged)
    if credentials:
        for field_name in ["api_key", "api_key_env", "base_url", "provider", "model"]:
            if not merged.get(field_name) and credentials.get(field_name):
                merged[field_name] = credentials[field_name]
        if isinstance(credentials.get("extra_body"), dict) and not merged.get("extra_body"):
            merged["extra_body"] = dict(credentials["extra_body"])
    if _string_or_empty(merged.get("provider", "")).lower() == "poe" and not merged.get("base_url"):
        merged["base_url"] = "https://api.poe.com/v1"
    if _string_or_empty(merged.get("provider", "")).lower() == "deepseek" and not merged.get("base_url"):
        merged["base_url"] = "https://api.deepseek.com"
    return apply_model_behavior_profile(merged)


def llm_models() -> list[dict[str, Any]]:
    models = CONFIG.get("llm_models")
    if isinstance(models, list) and models:
        return [apply_llm_credentials(dict(model)) for model in models]
    return []


def active_model_config() -> dict[str, Any]:
    models = llm_models()
    if not models:
        raise ValueError("No llm_models are configured.")
    if len(models) != 1:
        raise ValueError("Each experiment_config must contain exactly one llm_models entry for the experiment-ready workflow.")
    return models[0]


def llm_model_map() -> dict[str, dict[str, Any]]:
    return {str(model["key"]): model for model in llm_models()}


def active_llm_key() -> str:
    model_map = llm_model_map()
    configured = _string_or_empty(CONFIG.get("active_llm_key", "")).strip()
    if configured and configured in model_map:
        return configured
    if model_map:
        return next(iter(model_map))
    return ""


def llm_config_for(model_key: str | None = None) -> dict[str, Any]:
    model_map = llm_model_map()
    key = active_llm_key() if model_key is None else str(model_key)
    if key not in model_map:
        raise KeyError(f"Unknown llm model key: {key}")
    return apply_llm_credentials(dict(model_map[key]))


def ai_collection_dir_for(model_key: str | None = None) -> Path:
    if model_key is not None:
        llm_config_for(model_key)
    return AI_COLLECTION_DIR


def experiment_artifact_path(filename: str) -> Path:
    return EXPERIMENT_DIR / filename


def raw_output_path(filename: str) -> Path:
    return OUTPUT_DIR / filename


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_analysis_dir(base_dir: Path, family: str, dataset: str | None = None) -> Path:
    path = Path(base_dir) / str(family)
    if dataset is not None and str(dataset).strip():
        path = path / str(dataset)
    return ensure_dir(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def configured_indicator_names() -> list[str]:
    survey_config = CONFIG.get("survey_design", {})
    n_attitudes = int(survey_config.get("n_attitudes", len(INDICATOR_NAMES)))
    n_attitudes = max(0, min(n_attitudes, len(INDICATOR_NAMES)))
    return INDICATOR_NAMES[:n_attitudes]


def survey_total_tasks() -> int:
    survey_config = CONFIG.get("survey_design", {})
    component_total = sum(
        int(survey_config.get(key, 0))
        for key in [
            "n_core_tasks",
            "n_paraphrase_twins",
            "n_label_mask_twins",
            "n_order_twins",
            "n_monotonicity_tasks",
            "n_dominance_tasks",
        ]
    )
    if component_total > 0:
        return component_total
    return int(survey_config.get("total_tasks", 0))


def nested_response_value(payload: Any, path: str) -> Any:
    current = payload
    for part in str(path).split("."):
        token = part.strip()
        if not token:
            continue
        if isinstance(current, list):
            if not token.isdigit():
                return None
            index = int(token)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        if isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
            continue
        return None
    return current


def normalize_response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _number_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def response_decoder_config(model_config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "response_text_path": "choices.0.message.content",
        "thinking_text_path": "choices.0.message.reasoning_content",
        "done_reason_path": "choices.0.finish_reason",
        "total_duration_path": "",
        "prompt_eval_count_path": "usage.prompt_tokens",
        "eval_count_path": "usage.completion_tokens",
    }

    override = model_config.get("response_decoder", {})
    if isinstance(override, dict):
        for key, value in override.items():
            if value is not None:
                defaults[str(key)] = str(value)
    return defaults


def decode_chat_response(response: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    decoder = response_decoder_config(model_config)

    def read(path_key: str) -> Any:
        path = _string_or_empty(decoder.get(path_key, "")).strip()
        return nested_response_value(response, path) if path else None

    return {
        "response_text": normalize_response_text(read("response_text_path")).strip(),
        "thinking_text": normalize_response_text(read("thinking_text_path")).strip(),
        "metadata": {
            "done_reason": str(read("done_reason_path") or ""),
            "total_duration": _number_or_zero(read("total_duration_path")),
            "prompt_eval_count": _number_or_zero(read("prompt_eval_count_path")),
            "eval_count": _number_or_zero(read("eval_count_path")),
        },
    }


def default_api_key_env_names(model_config: dict[str, Any]) -> list[str]:
    provider = _string_or_empty(model_config.get("provider", "")).strip().lower()
    if provider == "poe":
        return ["POE_API_KEY"]
    if provider == "deepseek":
        return ["DEEPSEEK_API_KEY", "DeepSeek_API_KEY"]
    if provider in {"openai", "openai_compatible"}:
        return ["OPENAI_API_KEY"]
    return []


def resolve_llm_api_key(model_config: dict[str, Any]) -> str:
    api_key = _string_or_empty(model_config.get("api_key", "")).strip()
    if api_key:
        return api_key

    credential_data = load_credentials_payload(model_config)
    if credential_data:
        for key in ("api_key", "token", "poe_api_key", "POE_API_KEY", "deepseek_api_key", "DEEPSEEK_API_KEY", "DeepSeek_API_KEY"):
            candidate = _string_or_empty(credential_data.get(key, "")).strip()
            if candidate:
                return candidate

    api_key_env = _string_or_empty(model_config.get("api_key_env", "")).strip()
    if api_key_env:
        candidate = os.environ.get(api_key_env, "").strip()
        if candidate:
            return candidate

    for env_name in default_api_key_env_names(model_config):
        candidate = os.environ.get(env_name, "").strip()
        if candidate:
            return candidate

    return ""


def archive_experiment_config(trial_dir: Path | None = None) -> Path:
    target_dir = ensure_dir(EXPERIMENT_DIR if trial_dir is None else trial_dir)
    target = target_dir / "experiment_config.json"
    write_json(target, CONFIG)
    return target


def infer_trial_dir_from_output_dir(output_dir: Path) -> Path | None:
    for parent in [output_dir, *output_dir.parents]:
        if parent.name == "outputs":
            return parent.parent
    return None


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_json_payload(text: str) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_indicator_value(text: str) -> int:
    payload = parse_json_payload(text)
    for key in ("indicator_value", "value", "response"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError):
                value = -1
            return value if value in VALID_INDICATOR_VALUES else -1
    match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else -1


def parse_choice_label(text: str) -> str:
    payload = parse_json_payload(text)
    for key in ("choice_label", "choice", "answer"):
        if key in payload:
            label = str(payload[key]).strip().upper()
            if label in CHOICE_LABEL_TO_CODE:
                return label
            if label in {"PT", "CAR", "SLOW_MODES"}:
                return {"PT": "A", "CAR": "B", "SLOW_MODES": "C"}[label]
    match = re.search(r"\b([A-C])\b", text.upper())
    if match:
        return match.group(1)
    for name, label in {"PT": "A", "CAR": "B", "SLOW_MODES": "C"}.items():
        if re.search(rf"\b{name}\b", text.upper()):
            return label
    return ""


def parse_task_response(text: str) -> dict[str, Any]:
    payload = parse_json_payload(text)
    result = {
        "choice_label": "",
        "confidence": -1,
        "top_attributes": [],
        "dominated_option_seen": None,
    }
    if not payload:
        result["choice_label"] = parse_choice_label(text)
        confidence_match = re.search(r'"?confidence"?\s*[:=]\s*([1-5])', text, flags=re.IGNORECASE)
        if confidence_match:
            result["confidence"] = int(confidence_match.group(1))
        return result

    result["choice_label"] = parse_choice_label(json.dumps(payload, ensure_ascii=False))

    confidence = payload.get("confidence", -1)
    try:
        confidence = int(confidence)
    except (TypeError, ValueError):
        confidence = -1
    result["confidence"] = confidence if 1 <= confidence <= 5 else -1

    top_attributes = payload.get("top_attributes", [])
    if isinstance(top_attributes, str):
        top_attributes = [top_attributes]
    parsed_attributes: list[str] = []
    if isinstance(top_attributes, list):
        for item in top_attributes:
            label = str(item).strip().lower()
            if label in TASK_ATTRIBUTE_OPTIONS and label not in parsed_attributes:
                parsed_attributes.append(label)
            if len(parsed_attributes) == 2:
                break
    result["top_attributes"] = parsed_attributes

    dominated_option_seen = payload.get("dominated_option_seen", None)
    if isinstance(dominated_option_seen, bool):
        result["dominated_option_seen"] = dominated_option_seen
    elif isinstance(dominated_option_seen, str):
        lowered = dominated_option_seen.strip().lower()
        if lowered in {"true", "yes"}:
            result["dominated_option_seen"] = True
        elif lowered in {"false", "no"}:
            result["dominated_option_seen"] = False
    return result


def total_variation_distance(left: pd.Series, right: pd.Series) -> float:
    levels = sorted(set(left.index).union(set(right.index)))
    return 0.5 * sum(abs(float(left.get(level, 0.0)) - float(right.get(level, 0.0))) for level in levels)


def likert_probability_numpy(observed: np.ndarray, index_value: np.ndarray, delta_1: float, delta_2: float) -> np.ndarray:
    tau_1 = delta_1
    tau_2 = delta_1 + delta_2
    thresholds = np.array([-tau_2, -tau_1, 0.0, tau_1, tau_2], dtype=float)
    upper = np.empty_like(index_value)
    lower = np.empty_like(index_value)
    observed_zero = observed.astype(int) - 1
    for category in range(6):
        mask = observed_zero == category
        if not np.any(mask):
            continue
        upper_cut = thresholds[category] if category < 5 else np.inf
        lower_cut = thresholds[category - 1] if category > 0 else -np.inf
        upper[mask] = np.where(np.isfinite(upper_cut), norm.cdf(upper_cut - index_value[mask]), 1.0)
        lower[mask] = np.where(np.isfinite(lower_cut), norm.cdf(lower_cut - index_value[mask]), 0.0)
    return np.clip(upper - lower, 1e-30, 1.0)
