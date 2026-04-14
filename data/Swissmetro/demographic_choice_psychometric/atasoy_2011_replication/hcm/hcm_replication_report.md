# Atasoy 2011 continuous HCM replication

This report fixes the normalization to `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude.

The strict reproduction rule is simple: for each parameter or summary quantity explicitly reported in the paper, our estimate must match the paper after rounding to the same number of displayed decimals.

Rounded matches: `1` / `36`. Choice-only log-likelihood: `-1067.477` versus paper `-1069.8`.

| Name | Paper | Ours | Status |
| --- | ---: | ---: | --- |
| ASCPMM | -0.599 | -0.598 | mismatch |
| ASCSM | -0.772 | -0.771 | mismatch |
| beta_cost | -0.0559 | -0.0577 | mismatch |
| beta_time_pmm | -0.0294 | -0.0298 | mismatch |
| beta_time_pt | -0.0119 | -0.0122 | mismatch |
| beta_distance | -0.224 | -0.225 | mismatch |
| beta_ncars | 0.970 | 0.975 | mismatch |
| beta_nchildren | 0.215 | 0.204 | mismatch |
| beta_language | 1.060 | 1.054 | mismatch |
| beta_work | -0.583 | -0.582 | mismatch |
| beta_urban | 0.283 | 0.283 | match |
| beta_student | 3.260 | 3.261 | mismatch |
| beta_nbikes | 0.385 | 0.376 | mismatch |
| beta_Acar | -0.574 | -0.565 | mismatch |
| beta_Aenv | 0.393 | 0.392 | mismatch |
| Acar | 3.020 | 2.964 | mismatch |
| Aenv | 3.230 | 3.295 | mismatch |
| theta_ncars | 0.1040 | 0.1091 | mismatch |
| theta_educ | 0.2350 | 0.2285 | mismatch |
| theta_nbikes | 0.0845 | 0.0712 | mismatch |
| theta_age | 0.00445 | 0.00380 | mismatch |
| theta_valais | -0.2230 | -0.2127 | mismatch |
| theta_bern | -0.3610 | -0.3851 | mismatch |
| theta_basel_zurich | -0.2560 | -0.2312 | mismatch |
| theta_east | -0.2280 | -0.2252 | mismatch |
| theta_graubunden | -0.3030 | -0.2866 | mismatch |
| choice_log_likelihood | -1069.8 | -1067.5 | mismatch |
| market_share_PMM | 0.6311 | 0.6290 | mismatch |
| market_share_PT | 0.3120 | 0.3154 | mismatch |
| market_share_SM | 0.0569 | 0.0556 | mismatch |
| elasticity_PMM_cost | -0.058 | -0.061 | mismatch |
| elasticity_PMM_time | -0.234 | -0.241 | mismatch |
| elasticity_PT_cost | -0.202 | -0.209 | mismatch |
| elasticity_PT_time | -0.465 | -0.474 | mismatch |
| value_of_time_PMM | 31.54 | 31.03 | mismatch |
| value_of_time_PT | 12.81 | 12.65 | mismatch |
