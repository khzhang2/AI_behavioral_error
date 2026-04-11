# AI Behavioral Error

This repository is now a lightweight academic codebase centered on the Optima dataset. The current active methodological line is the Optima latent response regime experiment, while earlier retained Optima hybrid choice model and multinomial logit archives remain in `experiments/Swissmetro` as archival outputs. The codebase is intentionally organized in a simple manner: the root `experiment_config.json` stores the active configuration, the root `data/` directory stores both the retained source data and the newly generated experiment-specific data packages, the root `scripts/` directory stores all maintained scripts in a flat structure, and the `experiments/` directory is used only to archive outputs.

The present repository supports multiple artificial-intelligence respondent backends through the `llm_models` block in `experiment_config.json`. Each entry in that list defines one callable model, including its provider, model identifier, collection subdirectory, decoding settings, and any credential-related fields. The field `active_llm_key` selects the default model for scripts that use a single active respondent backend, while scripts that explicitly accept a model key can run several models side by side. At present, the repository supports local `ollama`, generic OpenAI-compatible endpoints, and Poe’s OpenAI-compatible application programming interface (API).

The maintained Optima scripts are:

- `prepare_optima_data.py`
- `run_optima_ai_collection.py`
- `estimate_optima_biogeme_hcm.py`
- `estimate_optima_torch_hcm.py`
- `estimate_optima_biogeme_mnl.py`
- `prepare_optima_latent_regime_data.py`
- `optima_latent_regime_questionnaire.py`
- `run_optima_latent_regime_ai_collection.py`
- `estimate_optima_panel_mnl.py`
- `estimate_optima_salcm.py`
- `summarize_optima_latent_regime.py`
- `optima_common.py`

For routine work on Windows, the expected interpreter is:

```powershell
.\.venv\Scripts\python.exe
```

## Using Poe as the artificial-intelligence respondent backend

Poe is supported through the same `llm_models` mechanism used for other respondent backends. In practical terms, a Poe-backed model entry is defined in `experiment_config.json` by setting `provider` to `"poe"` and by supplying a Poe model name in the `model` field. The OpenAI-compatible Poe endpoint is `https://api.poe.com/v1`, and the repository will use that endpoint when `provider = "poe"`.

The most convenient way to switch one experiment line to Poe is to add or edit one model entry in `llm_models`, then set `active_llm_key` to that model key if the intended script uses a single active model. A typical Poe model entry has the following structure:

```json
{
  "key": "poe_deepseek_r1_8b",
  "collection_subdir": "ai_collection_poe_deepseek_r1_8b",
  "respondent_prefix": "PO",
  "provider": "poe",
  "model": "deepseek-r1:8b",
  "base_url": "https://api.poe.com/v1",
  "api_key_env": "POE_API_KEY",
  "api_key": "",
  "api_key_file": "poe_api_credentials.local.json",
  "format": "json",
  "temperature": 0.1,
  "top_p": 0.95,
  "top_k": 20
}
```

Three credential routes are supported. The script first checks whether `api_key` is directly present in the model entry, then whether a local credential file is specified through `api_key_file`, and finally whether an environment variable is specified through `api_key_env`. In ordinary use, the safest route is the local credential file, because it keeps the token out of the tracked configuration file while remaining simple to edit on one machine.

The local Poe credential file is:

- `poe_api_credentials.local.json`

Its expected structure is:

```json
{
  "poe": {
    "api_key": "",
    "base_url": "https://api.poe.com/v1"
  }
}
```

This file is ignored by Git. Therefore, you can place the Poe token in the `api_key` field without creating a tracked secret in the repository. If you prefer environment variables instead, you may leave the file empty and set `POE_API_KEY` in the local shell environment.

## Local files and directories used by the repository

The repository now uses a small number of local files and directories that are important for day-to-day experimentation.

- `experiment_config.json` is the active root configuration. Before a run, this file determines the active data package, the archive directory, the available respondent backends, and the estimation settings. During collection and estimation, the active configuration is copied into the corresponding trial archive under `experiments/` without overwriting earlier copies.
- `poe_api_credentials.local.json` is the optional local credential store for Poe. It is intended for machine-specific sensitive tokens.
- `data/Swissmetro/demographic_choice_psychometric` stores the retained Optima source package, including the cleaned human benchmark files and the earlier archived artificial-intelligence collections.
- `data/Swissmetro/latent_regime_optima_v1` stores the new latent response regime experiment package, including the scenario bank, respondent-profile bank, block assignments, panel tasks, and the new artificial-intelligence collections for each configured model.
- `experiments/Swissmetro/...` stores archival outputs only. Each trial directory contains estimation outputs, summaries, and one or more archived copies of the active `experiment_config.json`.
- `biogeme_runtime.toml` is a local runtime file written when Biogeme-based estimators are run.
- `*.iter` files are Biogeme iteration snapshots. They are not the main experimental outputs, but may appear in the repository root when Biogeme writes optimization checkpoints.

Within the new latent response regime line, each model-specific collection directory under `data/Swissmetro/latent_regime_optima_v1` contains:

- `persona_samples.csv`
- `parsed_attitudes.csv`
- `parsed_task_responses.csv`
- `ai_panel_long.csv`
- `ai_panel_block.csv`
- `raw_interactions.jsonl`
- `respondent_transcripts.json`
- `run_respondents.json`

These files together define the complete survey-side record of one artificial-intelligence respondent backend.

## Workflow summary

The intended workflow remains deliberately direct. First prepare the data package needed for the active experiment. Second, run the questionnaire collection script if a new artificial-intelligence respondent survey is required. Third, estimate the benchmark and artificial-intelligence models. Finally, review the archived outputs in `experiments/Swissmetro/...`.

The code intentionally favors readability over software-engineering abstraction. The scripts are therefore written as direct research scripts, with limited indirection, limited logging, and explicit file paths.
