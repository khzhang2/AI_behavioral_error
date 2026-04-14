# AI Behavioral Error — AI Coding Assistant Guide

> For project overview, setup, directory structure, configuration, workflow, and LLM backend details, see [README.md](README.md).

## Scope

This file contains conventions **specific to AI coding assistants** working in this repository. It supplements — and intentionally does not duplicate — the README.

## Python Environment

- `./.venv/bin/python` (Mac/Linux)
- `.\.venv\Scripts\python.exe` (Windows)

## Repo-local Skill

Post-AI analysis only: [optima-experiment-workflow](.codex/skills/optima-experiment-workflow/SKILL.md). Use it to validate collection completion and run analysis; **do not** use it to launch AI questionnaire collection.

## Key Rules

1. **One model per experiment folder.** Each experiment-ready config must have exactly one `llm_models` entry.
2. **Use `--model-key`, not `model`.** The `key` field is the internal unique ID; `model` is the backend name.
3. **Do not mix raw and derived outputs.** `outputs/` stores only raw AI collection files. Diagnostics, panels, and estimation results go in the experiment root or named sub-directories (`atasoy_2011_replication/`, `hcm/`, `salcm/`).
4. **Re-run prepare after config changes.** If `survey_design`, `n_block_templates_per_model`, or `n_repeats_per_template` change, run `prepare_optima_intervention_regime_data.py` before collection.
5. **Config priority:** `experiment_config.json` overrides → `experiment_config_base.json`. Within a model entry, explicit fields override `api_credentials.local.json`.
6. **Write experiment summaries in Chinese** unless the user specifies otherwise.
7. **Sobol draws are shared.** Pre-generated `.npy` files ensure identical Monte Carlo integration across methods.
8. **Gitignored runtime files:** `*.iter`, `biogeme.toml`, `biogeme_runtime.toml`, `api_credentials.local.json`, `raw_interactions.jsonl`, `respondent_transcripts.json`.
9. **Keep Atasoy AI analysis on the human code path.** `estimate_atasoy_2011_ai_analysis.py` should reorganize AI outputs into the same Atasoy-style estimation table used by `replicate_atasoy_2011_models.py`, and then reuse the same base-logit and exact-HCM estimation functions.

## Active Pipeline Scripts

See [README.md](README.md) for the full list of active scripts.


## LLM Backend Notes

- **Poe models:** keep `provider`, `model`, `base_url`, `credentials_file` in `llm_models`. Store only `api_key` / `api_key_env` in `api_credentials.local.json`.
- **MLX local models:** use `provider = "mlx"` and set `model` to a `mlx_lm.load()`-compatible Hugging Face MLX repo id. The current default small-model key is `mlx_qwen35_0p8b` with model `mlx-community/Qwen3.5-0.8B-5bit`.
- **Keep MLX config shape stable.** Even when MLX does not use a field such as `base_url`, `credentials_file`, `format`, `reasoning_effort`, `top_k`, `timeout_sec`, `extra_body`, or `response_decoder`, prefer keeping that field in `llm_models` with an empty placeholder value rather than dropping the key.
- **MLX collection is single-worker only.** Even if `collection.max_workers` is set larger, the collection script will warn and run with one worker.
- If `api_key` is empty, the code falls back to the system environment variable (e.g. `POE_API_KEY`, `DEEPSEEK_API_KEY`).
- Use `response_decoder` in `llm_models` to override default output parsing paths when the backend deviates from the standard format.
