# AI Behavioral Error - Project Guide

## Project Overview

This is an academic research codebase studying whether LLM-generated synthetic respondents can reproduce human behavioral patterns in discrete choice models. The project uses the Optima dataset (Swissmetro travel survey) as the benchmark, collects AI survey responses via LLMs, and compares human and AI behavior using intervention diagnostics, panel Multinomial Logit (MNL), and SALCM.

Legacy HCM estimation files are retained for reference only and are not part of the current default experiment workflow.

Python runtime: `.\.venv\Scripts\python.exe` (Windows)

Repo-local skill for post-AI analysis only: [optima-experiment-workflow](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/.codex/skills/optima-experiment-workflow/SKILL.md). Use it to validate collection completion and run analysis; do not use it to launch AI questionnaire collection.

## Directory Structure

```
AI_behavioral_error/
├── data/                          # All datasets (human + AI collected)
│   └── Swissmetro/
│       └── demographic_choice_psychometric/
│           ├── raw/               # Original Optima .dat file from Biogeme distribution
│           ├── ai_collection_qwen3.5_9b/    # AI survey outputs (Qwen 3.5 9B)
│           ├── ai_collection_deepseek_r1_8b/ # AI survey outputs (DeepSeek R1 8B)
│           ├── human_cleaned_wide.csv        # Cleaned human data (wide format)
│           ├── human_cleaned_long.csv        # Cleaned human data (long format)
│           ├── human_respondent_profiles.csv # Human respondent profiles
│           ├── shared_sobol_draws_32.npy     # Shared Sobol draws (32 draws)
│           ├── shared_sobol_draws_500.npy    # Shared Sobol draws (500 draws)
│           ├── human_benchmark_sample_summary.json
│           ├── optima_codebook.json
│           └── optima_data_description.md
├── scripts/                       # All runnable scripts (flat structure)
├── experiments/                   # Archived experiment outputs (read-only after generation)
│   └── Swissmetro/
│       └── YYYYMMDD_<keywords>_<version>/
├── docs/                          # Documentation and audit notes
├── experiment_config.json         # Active experiment overrides
├── experiment_config_base.json    # Reusable experiment base template
├── biogeme.toml                   # Biogeme default parameter file (gitignored)
├── biogeme_runtime.toml           # Biogeme runtime overrides (written by scripts, gitignored)
├── pyproject.toml                 # Python project metadata and dependencies
└── AGENTS.md                      # This file
```

### `scripts/`

All runnable Python scripts in a flat structure. No sub-packages. Scripts import from each other directly:

### experiments

Archived outputs only. Each experiment run is stored under `experiments/Swissmetro/YYYYMMDD_<keywords>_<model_name>_<version>/`. Each experiment folder must correspond to exactly one model. Contains:

- `experiment_config.json` - the single full final config used for that experiment.
- `outputs/` - only raw AI collection files such as `raw_interactions.jsonl`, `respondent_transcripts.json`, `run_respondents.json`, and `ai_collection_summary.json`.
- experiment-root files - shared derived AI panels, shared diagnostics, questionnaire-construction outputs, and one `experiment_summary.md`.
- `hcm/ai`, `hcm/human`, `mnl/ai`, `mnl/human`, `salcm/ai`, `salcm/human` - model-specific estimation inputs and outputs.


## Key Conventions

- All paths are resolved relative to the project root via `optima_common.ROOT_DIR = Path(__file__).resolve().parents[1]`.
- The active experiment is determined by `experiment_config.json` plus `experiment_config_base.json`. The overrides in `experiment_config.json` take precedence.
- Each experiment-ready config must contain exactly one `llm_models` entry.
- AI collection supports resume via `--resume` flag; it replays `raw_interactions.jsonl` to rebuild state.
- Invalid LLM responses trigger a single repair attempt before moving on.
- Sobol draws are pre-generated and shared across all estimation methods to ensure identical Monte Carlo integration.
- Biogeme `*.iter` files and `biogeme.toml` / `biogeme_runtime.toml` are gitignored; the runtime TOML is rewritten on each Biogeme estimation run.
- `experiment_config.json` now keeps only tuning/active overrides and points to a reusable base template through `config_base_file`.
- The complete experiment definition is now stored in `experiment_config_base.json`; `paths.archive_dir` is the parent archive directory, and the real experiment folder is `paths.archive_dir / experiment_name`. The final merged version is written once into that experiment folder's `experiment_config.json` during `archive_experiment_config` calls.
- Experiment-level knobs such as `active_llm_key`, `llm_models`, `n_block_templates_per_model`, `n_repeats_per_template`, and `survey_design` should stay in `experiment_config.json`.
- 对于 Poe 模型，更推荐继续在 `llm_models` 中保留统一格式：
  - `provider`
  - `model`
  - `base_url`
  - `credentials_file`
- `api_credentials.local.json` 只保留 `api_key` 或 `api_key_env`。
- 如果 `api_key` 为空，代码会自动尝试从系统环境变量 `POE_API_KEY` 读取。
- 如果不同模型的输出字段结构不同，可在 `llm_models` 中提供 `response_decoder` 来覆盖默认解析路径。
