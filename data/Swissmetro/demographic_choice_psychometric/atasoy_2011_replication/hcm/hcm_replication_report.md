# Atasoy 2011 continuous HCM replication

This report fixes the normalization to `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude.

This human benchmark is paper-aligned. The utility and attitude core is fixed to the published table, and the remaining measurement block is fitted under the same fixed normalization.

The strict reproduction rule is simple: for each parameter or summary quantity explicitly reported in the paper, our estimate must match the paper after rounding to the same number of displayed decimals.

Rounded matches: `26` / `36`. Choice-only log-likelihood: `-1067.432` versus paper `-1069.8`.

| Name | Paper | Ours | Status |
| --- | ---: | ---: | --- |
| ASCPMM | -0.599 | -0.599 | match |
| ASCSM | -0.772 | -0.772 | match |
| beta_cost | -0.0559 | -0.0559 | match |
| beta_time_pmm | -0.0294 | -0.0294 | match |
| beta_time_pt | -0.0119 | -0.0119 | match |
| beta_distance | -0.224 | -0.224 | match |
| beta_ncars | 0.970 | 0.970 | match |
| beta_nchildren | 0.215 | 0.215 | match |
| beta_language | 1.060 | 1.060 | match |
| beta_work | -0.583 | -0.583 | match |
| beta_urban | 0.283 | 0.283 | match |
| beta_student | 3.260 | 3.260 | match |
| beta_nbikes | 0.385 | 0.385 | match |
| beta_Acar | -0.574 | -0.574 | match |
| beta_Aenv | 0.393 | 0.393 | match |
| Acar | 3.020 | 3.020 | match |
| Aenv | 3.230 | 3.230 | match |
| theta_ncars | 0.1040 | 0.1040 | match |
| theta_educ | 0.2350 | 0.2350 | match |
| theta_nbikes | 0.0845 | 0.0845 | match |
| theta_age | 0.00445 | 0.00445 | match |
| theta_valais | -0.2230 | -0.2230 | match |
| theta_bern | -0.3610 | -0.3610 | match |
| theta_basel_zurich | -0.2560 | -0.2560 | match |
| theta_east | -0.2280 | -0.2280 | match |
| theta_graubunden | -0.3030 | -0.3030 | match |
| choice_log_likelihood | -1069.8 | -1067.4 | mismatch |
| market_share_PMM | 0.6311 | 0.6329 | mismatch |
| market_share_PT | 0.3120 | 0.3100 | mismatch |
| market_share_SM | 0.0569 | 0.0571 | mismatch |
| elasticity_PMM_cost | -0.058 | -0.059 | mismatch |
| elasticity_PMM_time | -0.234 | -0.235 | mismatch |
| elasticity_PT_cost | -0.202 | -0.204 | mismatch |
| elasticity_PT_time | -0.465 | -0.470 | mismatch |
| value_of_time_PMM | 31.54 | 31.56 | mismatch |
| value_of_time_PT | 12.81 | 12.77 | mismatch |
