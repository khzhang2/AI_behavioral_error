# 20260415_optima_intervention_regime_poe_gpt54_nano_v1 Atasoy 2011 analysis

This note applies the same Atasoy 2011 base logit and fixed-normalization continuous HCM estimation code used by the human replication to the AI outputs in this experiment.

The AI estimation input is first reorganized into the same Atasoy-style row format as the human replication. The current sample contains `2400` core-task observations from `400` completed AI respondents.

## Base logit

| Metric | Human paper replication | This AI experiment |
| --- | ---: | ---: |
| log-likelihood | -1067.356 | -918.702 |
| PMM VOT (CHF/hour) | 30.35 | 10.81 |
| PT VOT (CHF/hour) | 12.24 | 11.39 |
| PMM share | 0.6231 | 0.8409 |
| PT share | 0.3209 | 0.1484 |
| SM share | 0.0560 | 0.0107 |

## Exact HCM

The AI exact HCM uses the same fixed normalization and the same model equations as the human benchmark: `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude. The human side is stored as a paper-aligned canonical benchmark, while the AI side is estimated with the repository local-basin optimizer under that same normalization.

| Metric | Human HCM | This AI experiment |
| --- | ---: | ---: |
| choice-only log-likelihood | -1067.432 | -991.524 |
| PMM VOT (CHF/hour) | 31.56 | 8.66 |
| PT VOT (CHF/hour) | 12.77 | 9.97 |
| mean Acar | 2.886 | 2.954 |
| mean Aenv | 3.547 | 3.711 |
