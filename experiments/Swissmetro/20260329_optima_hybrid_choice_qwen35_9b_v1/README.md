# Optima Reduced-Form Archive Note

This archive directory stores the outputs of the retained Optima hybrid choice exercise based on `qwen3.5:9b`. In the current repository organization, development no longer occurs inside `experiments/`. The maintained code now lives in the root `scripts/` directory, and the active data package lives in the root `data/Swissmetro/demographic_choice_psychometric` directory. This folder should therefore be read as an archive of one completed trial rather than as the development location.

The archived outputs in this folder correspond to the reduced official-style hybrid choice model that combines two latent variables, six psychometric indicators, and one mode-choice equation. The main estimation outputs are stored under `outputs/human_biogeme_32`, `outputs/ai_biogeme_32`, `outputs/human_torch_32`, `outputs/ai_torch_32`, `outputs/human_torch_500`, and `outputs/ai_torch_500`. Comparison files are stored under `outputs/aggregate`.

The active maintained scripts corresponding to this archive are now:

- `scripts/prepare_optima_data.py`
- `scripts/run_optima_ai_collection.py`
- `scripts/estimate_optima_biogeme_hcm.py`
- `scripts/estimate_optima_torch_hcm.py`
- `scripts/compare_optima_hcm.py`
- `scripts/estimate_optima_biogeme_mnl.py`
- `scripts/summarize_optima_experiment.py`

The root `experiment_config.json` controls which data package and archive directory are currently active. As a result, if this archived experiment needs to be reproduced, the correct procedure is to point the root configuration back to this archive path and then run the corresponding root-level scripts.
