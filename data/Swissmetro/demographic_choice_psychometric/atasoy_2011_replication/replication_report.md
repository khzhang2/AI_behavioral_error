# Atasoy 2011 replication

This directory is the canonical human replication of the Atasoy, Glerum, and Bierlaire (2011) base logit model and continuous hybrid choice model.

The replication sample keeps all observations with `Choice != -1`, which gives `1906` loop observations from the public `optima.dat`.

The base logit outputs are saved under `/Users/kaihangzhang/Downloads/GitHub/Research codes repo/AI_behavioral_error/data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/base_logit`. Rounded to the paper precision, `18` of `23` literature-reported base quantities match.

The continuous hybrid choice outputs are saved under `/Users/kaihangzhang/Downloads/GitHub/Research codes repo/AI_behavioral_error/data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/hcm`. The repository now uses the fixed normalization `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude. Rounded to the paper precision, `1` of `36` literature-reported continuous-model quantities match.

Re-run command:

```bash
./.venv/bin/python scripts/replicate_atasoy_2011_models.py --output-dir "/Users/kaihangzhang/Downloads/GitHub/Research codes repo/AI_behavioral_error/data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication"
```
