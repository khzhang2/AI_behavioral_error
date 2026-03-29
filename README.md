# AI Behavioral Error

This repository is now a lightweight academic codebase centered on the Optima dataset. The repository therefore keeps one active experiment family, namely the Optima hybrid choice and reduced multinomial logit (MNL) exercises.

The repository is organized in a deliberately simple manner. The root `experiment_config.json` stores the active configuration. The root `data/` directory stores the Optima data package and the processed human benchmark files. The root `scripts/` directory stores all maintained scripts in a flat structure, without experiment-specific subfolders. The `experiments/` directory is used only to archive outputs.

The active large language model (LLM) settings are defined in the `llm` block of `experiment_config.json`. The same configuration file also stores the active data path, the archive path, the Biogeme settings, and the torch settings. Local `ollama` and OpenAI-compatible application programming interface (API) endpoints are both supported through these fields, although the current active configuration uses `qwen3.5:9b` through `ollama`.

The maintained Optima scripts are:

- `prepare_optima_data.py`
- `run_optima_ai_collection.py`
- `optima_questionnaire_template.py`
- `estimate_optima_biogeme_hcm.py`
- `estimate_optima_torch_hcm.py`
- `compare_optima_hcm.py`
- `estimate_optima_biogeme_mnl.py`
- `summarize_optima_experiment.py`
- `optima_common.py`
- `optima_hcm_model_spec.py`

The intended workflow is straightforward. First prepare the data if necessary. Then run the questionnaire collection script if a new synthetic respondent survey is needed. After that, estimate either the hybrid choice model or the reduced MNL model using the archived Optima outputs. For routine work on Windows, the expected interpreter is:

```powershell
.\.venv\Scripts\python.exe
```

The code intentionally favors readability over software-engineering abstraction. The scripts are therefore written as direct research scripts, with limited indirection, limited logging, and explicit file paths.
