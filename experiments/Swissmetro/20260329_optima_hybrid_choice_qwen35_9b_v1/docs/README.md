# Optima Reduced Official-Style Hybrid Choice Replication

This directory contains a lightweight research pipeline for an Optima hybrid choice model under `experiments/Swissmetro`.

## Main idea

- official public `optima.dat`
- reduced official-style two latent variable HCM
- `qwen3.5:9b` via local `ollama`
- compact grounding
- one growing conversation per respondent
- `Biogeme32 -> torch32 alignment -> torch500`

## Main files

- `data/raw/optima.dat`
- `scripts/prepare_optima_data.py`
- `scripts/questionnaire_template.py`
- `scripts/run_ai_collection.py`
- `scripts/estimate_biogeme_hcm.py`
- `scripts/estimate_torch_hcm.py`
- `scripts/build_comparison.py`
- `scripts/summarize_experiment.py`

## Recommended run order

1. Prepare official data and shared Sobol draws

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\prepare_optima_data.py
```

2. Smoke-test AI survey with one respondent

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\run_ai_collection.py --n-respondents 1
```

3. Run full AI collection

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\run_ai_collection.py
```

4. Biogeme 32-draw estimation

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\estimate_biogeme_hcm.py --dataset human --n-draws 32 --output-subdir human_biogeme_32
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\estimate_biogeme_hcm.py --dataset ai --n-draws 32 --output-subdir ai_biogeme_32
```

5. Torch 32-draw alignment runs

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\estimate_torch_hcm.py --dataset human --n-draws 32 --output-subdir human_torch_32
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\estimate_torch_hcm.py --dataset ai --n-draws 32 --output-subdir ai_torch_32
```

6. If alignment is acceptable, run torch 500-draw estimation

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\estimate_torch_hcm.py --dataset human --n-draws 500 --output-subdir human_torch_500 --start-values outputs\human_torch_32\torch_hcm_estimates.csv
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\estimate_torch_hcm.py --dataset ai --n-draws 500 --output-subdir ai_torch_500 --start-values outputs\ai_torch_32\torch_hcm_estimates.csv
```

7. Build comparisons and summary

```powershell
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\build_comparison.py
python experiments\Swissmetro\20260329_optima_hybrid_choice_qwen35_9b_v1\scripts\summarize_experiment.py
```

## Notes

- The implemented HCM is reduced official-style, not the full official 75-parameter specification.
- The selected indicator statements are taken from the public Optima description when recoverable.
- Biogeme uses `cpu_count - 1` threads.
