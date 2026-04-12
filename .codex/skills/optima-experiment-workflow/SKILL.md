---
name: optima-experiment-workflow
description: Plan or execute the end-to-end Optima intervention-regime experiment workflow in this repository. Use when Codex needs to set or review experiment_config.json, create a new single-model experiment folder, generate personas and choice cards before AI collection, run or resume AI questionnaire collection, estimate intervention metrics or discrete-choice models after collection, interpret the five error dimensions, or enforce this repo's experiment record rules.
---

# Optima Experiment Workflow

## Overview

Use this skill for the current experiment-ready workflow in this repository. Treat the workflow as four linked stages: parameter design in `experiment_config.json`, pre-AI preparation of personas and questionnaire tasks, AI questionnaire collection, and post-AI estimation plus experiment summaries.

Use the current intervention-regime line as the default workflow. Treat older hybrid-choice or legacy multi-output layouts as reference only unless the user explicitly asks for them.

## Define the main objects first

- `task`: one question. It can be one attitude question or one choice card.
- `block template`: one reusable questionnaire-session template.
- `respondent block`: one `block template` under fixed experimental conditions such as model, prompt arm, persona, and temperature.
- `run`: one full execution of one `respondent block`.

In this repository, one full questionnaire run means:

`1 grounding + n_attitudes + total task cards`

## Fix the experiment target before doing anything else

Work with one experiment folder and one model at a time.

- Use the naming rule `YYYYMMDD_<keywords>_<version>`.
- Keep one experiment folder for one model only.
- Keep `llm_models` in `experiment_config.json` to one entry only for the active experiment-ready workflow.
- Archive the final merged configuration into the experiment folder's single `experiment_config.json`.

If the user changes `n_block_templates_per_model`, `n_repeats_per_template`, or anything in `survey_design`, run the prepare script again before running AI collection. The collection script reads the already-generated `block_assignments.csv` and `panel_tasks.csv`; it does not regenerate them from config on the fly.

## Edit parameters in the right file

Use `experiment_config.json` as the main tuning file. Treat `experiment_config_base.json` as stable defaults.

The fields most likely to change between experiments are:

- `experiment_name`
- `paths.archive_dir`
- `active_llm_key`
- `llm_models[0]`
- `n_block_templates_per_model`
- `n_repeats_per_template`
- `collection.max_workers`
- `survey_design`

When the provider is Poe, keep the model entry in `llm_models`. Put `provider`, `model`, `base_url`, sampling parameters, and decoder settings there. Keep only the secret in `api_credentials.local.json` or the environment variable such as `POE_API_KEY`.

To count expected requests, use:

`total requests = n_block_templates_per_model × n_repeats_per_template × (1 + n_attitudes + total task cards)`

And use:

`total task cards = n_core_tasks + n_paraphrase_twins + n_label_mask_twins + n_order_twins + n_monotonicity_tasks + n_dominance_tasks`

## Run the pre-AI prepare stage

Run:

```bash
./.venv/bin/python scripts/prepare_optima_intervention_regime_data.py --model-key <model_key>
```

This stage does the following.

1. Read the human profile bank and scenario bank.
2. Construct personas from human profiles.
3. Construct core choice cards and twin or probe cards from scenarios and `survey_design`.
4. Expand planned runs into `block_assignments.csv`.
5. Expand all task cards into `panel_tasks.csv`.
6. Initialize the experiment folder and the raw AI output files under `outputs/`.

Treat these files as the key pre-AI artifacts in the experiment root:

- `scenario_bank.csv`
- `respondent_profile_bank.csv`
- `block_assignments.csv`
- `panel_tasks.csv`
- `experiment_config.json`

## Run the AI questionnaire collection

Run:

```bash
./.venv/bin/python scripts/run_optima_intervention_regime_ai_collection.py --model-key <model_key> --max-workers <N>
```

Or resume:

```bash
./.venv/bin/python scripts/run_optima_intervention_regime_ai_collection.py --model-key <model_key> --max-workers <N> --resume
```

Use these rules when reasoning about performance and restart behavior.

- Keep one respondent serial inside a run: `grounding -> attitudes -> tasks`.
- Allow different respondents to run in parallel.
- Treat `--max-workers` as respondent-level parallelism, not task-level parallelism.
- Expect low CPU usage when the bottleneck is the remote API rather than local computation.

The intervention collection script writes incrementally.

- Each grounding, attitude, or task response is appended immediately to `outputs/raw_interactions.jsonl`.
- Parsed rows are appended immediately to `parsed_attitudes.csv` and `parsed_task_responses.csv`.
- `outputs/respondent_transcripts.json` and `outputs/run_respondents.json` are updated during collection.

Because of this, `--resume` should continue from unfinished respondents and, within those respondents, from the next unfinished question when the needed rows already exist on disk.

## Know where the collection outputs go

Use this storage rule consistently.

`outputs/` stores raw AI collection artifacts only:

- `outputs/raw_interactions.jsonl`
- `outputs/respondent_transcripts.json`
- `outputs/run_respondents.json`
- `outputs/ai_collection_summary.json`

The experiment root stores all derived AI data and all estimation outputs:

- `persona_samples.csv`
- `parsed_attitudes.csv`
- `parsed_task_responses.csv`
- `ai_panel_long.csv`
- `ai_panel_block.csv`
- all diagnostics, MNL outputs, SALCM outputs, and summaries

## Run the post-AI estimation sequence

Use this order for the current intervention-regime workflow.

1. Estimate intervention and randomness metrics:

```bash
./.venv/bin/python scripts/estimate_optima_intervention_metrics.py
```

2. Estimate the human baseline panel multinomial logit:

```bash
./.venv/bin/python scripts/estimate_optima_panel_mnl.py --dataset human
```

3. Estimate the AI panel multinomial logit:

```bash
./.venv/bin/python scripts/estimate_optima_panel_mnl.py --dataset ai_pooled
```

4. Estimate the scale-adjusted latent class model:

```bash
./.venv/bin/python scripts/estimate_optima_salcm.py
```

5. Write the short experiment summary:

```bash
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

## Read the results using the five error dimensions

Use the following mapping when interpreting one completed experiment.

1. Random instability within one model:
   Read `exact_repeat_randomness.csv`, `intervention_metrics_summary.json`, and `outputs/ai_collection_summary.json`.
   Focus on `exact-repeat flip rate` and `response entropy`.

2. Semantic invariance:
   Read `intervention_sensitivity.csv` and `ai_panel_block.csv`.
   Focus on `paraphrase gap`, `paraphrase flip rate`, and whether the excess gap rises above the randomness baseline.

3. Label or order sensitivity:
   Read `intervention_sensitivity.csv` and `ai_panel_block.csv`.
   Compare `label gap` with `order gap`. Do not collapse them too early; one may be weak while the other is strong.

4. Trade-off fidelity:
   Read `ai_panel_block.csv` and `outputs/ai_collection_summary.json`.
   Focus on `dominance violation rate` and `monotonicity compliance rate`.

5. Human-relative distortion:
   Read `human_baseline_mnl_summary.json`, `ai_panel_mnl_summary.json`, and `ai_salcm_regime_summaries.csv`.
   Start with `choice share` differences. Treat `VOT/WTP` or elasticity-style interpretation more cautiously when the optimizer reports precision loss or iteration limits.

## Follow the experiment record rules

Keep the experiment archive clean.

- Do not mix multiple models in one experiment folder.
- Do not store diagnostics or estimation results under `outputs/`.
- Keep one root `experiment_summary.md` only.
- Keep `experiment_summary.md` short and decision-oriented.
- Prefer writing the summary after the main analysis scripts finish, not during collection.

When the user asks for a new experiment, create a new experiment folder name rather than reusing an old archive unless they explicitly want to overwrite or continue the old one.

## Write the experiment report in Chinese

Write `experiment_summary.md` in Chinese by default for this repository unless the user explicitly asks for another language.

Use a compact structure.

1. One short opening paragraph stating the experiment target, sample size, and the main overall conclusion.
2. One Markdown table summarizing the five error dimensions.
3. One short closing paragraph stating the main caveat, especially when the optimizer reports precision loss or iteration limits.

Use the five error dimensions as fixed report rows:

- 同一系统的随机不稳定性
- 对语义等价改写是否稳健
- 对标签或顺序是否过敏
- 是否真的在做 trade-off
- 是否只是总体像人

Use a Markdown table with columns of this form:

| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |
| --- | --- | --- | --- |

Keep the wording concrete. Report the actual metric values from the current experiment rather than generic placeholders. When possible, use the same metrics already produced by the pipeline, such as `exact-repeat flip rate`, `response entropy`, `paraphrase flip rate`, `label flip rate`, `order flip rate`, `monotonicity compliance rate`, `dominance violation rate`, and `share gap`.
