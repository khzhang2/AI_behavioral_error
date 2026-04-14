# 20260414_optima_intervention_regime_atasoy_qwen35_0p8b_smallfull_v1 Atasoy 2011 analysis

This note applies the same Atasoy 2011 base logit and fixed-normalization continuous HCM estimation code used by the human replication to the AI outputs in this experiment.

The AI estimation input is first reorganized into the same Atasoy-style row format as the human replication. The current sample contains `276` core-task observations from `46` completed AI respondents.

This is a partial-sample analysis run on `46` / `96` planned respondents because the collection was stopped early.

## Base logit

| Metric | Human paper replication | This AI experiment |
| --- | ---: | ---: |
| log-likelihood | -1067.356 | -0.000 |
| PMM VOT (CHF/hour) | 30.35 | 13.90 |
| PT VOT (CHF/hour) | 12.24 | 19.35 |
| PMM share | 0.6231 | 0.0000 |
| PT share | 0.3209 | 1.0000 |
| SM share | 0.0560 | 0.0000 |

## Exact HCM

The AI exact HCM uses the same fixed normalization and the same estimation code path as the human replication: `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude.

| Metric | Human HCM | This AI experiment |
| --- | ---: | ---: |
| choice-only log-likelihood | -1067.477 | -0.004 |
| PMM VOT (CHF/hour) | 31.03 | 2576.99 |
| PT VOT (CHF/hour) | 12.65 | 7717.22 |
| mean Acar | 2.849 | 2.403 |
| mean Aenv | 3.571 | 3.630 |
