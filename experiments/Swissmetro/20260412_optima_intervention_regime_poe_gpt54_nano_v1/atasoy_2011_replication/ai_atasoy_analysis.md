# 20260412_optima_intervention_regime_poe_gpt54_nano_v1 Atasoy 2011 analysis

This note applies the same Atasoy 2011 base logit and fixed-normalization continuous HCM estimation code used by the human replication to the AI outputs in this experiment.

The AI estimation input is first reorganized into the same Atasoy-style row format as the human replication. The current sample contains `2400` core-task observations from `400` completed AI respondents.

## Base logit

| Metric | Human paper replication | This AI experiment |
| --- | ---: | ---: |
| log-likelihood | -1067.356 | -779.660 |
| PMM VOT (CHF/hour) | 30.35 | 19.56 |
| PT VOT (CHF/hour) | 12.24 | 18.55 |
| PMM share | 0.6231 | 0.8666 |
| PT share | 0.3209 | 0.1277 |
| SM share | 0.0560 | 0.0058 |

## Exact HCM

The exact Atasoy 2011 continuous HCM is not feasible from these AI outputs.

Missing required indicators: Envir02, Envir06, Mobil10, Mobil11, Mobil16
