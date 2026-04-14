# 20260412_optima_intervention_regime_poe_gpt54_nano_v1 Atasoy 2011 analysis

This note applies the Atasoy, Glerum, and Bierlaire (2011) base logit specification to the existing AI core-choice outputs of this experiment without sending any new model requests.

## Base logit

The AI-side Atasoy-style base logit uses the six `core` tasks only. It keeps the paper utility structure and merges the missing socio-demographic controls from the original human source respondent linked through `human_id`.

| Metric | Human paper replication | This AI experiment |
| --- | ---: | ---: |
| log-likelihood | -1067.356 | -523.981 |
| PMM VOT (CHF/hour) | 30.35 | 33.06 |
| PT VOT (CHF/hour) | 12.24 | 27.81 |
| PMM share | 0.6231 | 0.8468 |
| PT share | 0.3209 | 0.1437 |
| SM share | 0.0560 | 0.0095 |

## Exact HCM feasibility

The exact Atasoy 2011 continuous HCM is not feasible from the current AI outputs.

Missing required indicators: Envir02, Envir06, Mobil10, Mobil11, Mobil16

The current intervention-regime AI survey collected only these six attitude questions:

Envir01, Envir05, LifSty01, LifSty07, Mobil05, Mobil12
