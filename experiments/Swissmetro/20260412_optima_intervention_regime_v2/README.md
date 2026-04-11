# Optima Intervention-Anchored Latent Regime Experiment V2

This archive records the second-stage intervention-anchored latent response regime experiment built on the retained Optima benchmark. The purpose of this experiment is methodological rather than purely descriptive. The experiment is designed to distinguish four objects that are often conflated in the audit of artificial-intelligence (AI) survey behavior: exact-repeat randomness, intervention sensitivity, latent response regimes, and behavioral distortion relative to a human benchmark.

The design keeps the Optima domain and the retained human benchmark data, but it changes the AI survey architecture. The experimental unit is a block template rather than a single task. Each block template fixes the sampled persona, the prompt family, the prompt arm, and a sixteen-task panel. The same template is then repeated as independent runs. This repeated-block structure allows the experiment to estimate a stochastic envelope under exact repetition before it asks whether controlled perturbations generate systematic response shifts.

The sixteen-task panel contains six core empirical tasks, two paraphrase twins, two label-mask twins, two order-randomization twins, two monotonicity diagnostics, and two dominance diagnostics. Each task returns a structured response containing the chosen option, confidence, the two most decisive attributes, and whether a clearly dominated option is perceived. In this way, the experiment remains suitable for Scale-Adjusted Latent Class Choice Model (SALCM) estimation while also generating the diagnostic measures needed for intervention-anchored identification.

The current archive contains a smoke-test run rather than a full experiment. The smoke test uses one block template and two independent repeats for each model, namely `qwen3.5:9b` and `deepseek-r1:8b`. This small run is not intended for substantive behavioral interpretation. Its purpose is to verify that the new identification sequence is operational from data preparation through summary output.

The smoke-test pipeline that has already been executed is the following. First, the new data package is created under `data/Swissmetro/latent_regime_optima_v2`. Second, repeated-task AI data are collected for both models. Third, the human benchmark Multinomial Logit (MNL) model is estimated on the retained Optima human data, and a pooled AI panel MNL is estimated on the smoke-test AI panel. Fourth, exact-repeat randomness and intervention gaps are computed before the SALCM is estimated. Fifth, the archive summary combines the stochastic-envelope results, the intervention diagnostics, the pooled panel MNL, and the SALCM regime summaries.

The key smoke-test outputs are stored in the following subdirectories.

- `outputs/human_baseline_mnl`
- `outputs/pooled_ai_panel_mnl`
- `outputs/intervention_diagnostics`
- `outputs/pooled_ai_salcm`
- `outputs/regime_diagnostics`

The main summary file is `outputs/experiment_summary.md`. The methodological design note that motivated this archive is stored at `docs/latent_regime_intervention_anchored_v2_draft.md`.

At the smoke-test scale, the identification logic is already visible. The exact-repeat flip rate is zero in this tiny sample, whereas the mean intervention gap is positive. However, the bootstrap interval for the excess intervention gap still includes zero, which is expected at such a small scale. In other words, the smoke test shows that the full identification framework is technically operational, but it does not yet provide enough statistical power to support a strong rejection of the pure stochastic-instability null.
