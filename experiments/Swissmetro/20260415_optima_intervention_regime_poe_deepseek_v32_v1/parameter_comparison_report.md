# deepseek-v3.2

这个文档由 `20260415_optima_intervention_regime_poe_deepseek_v32_v1` 下的参数对照表自动生成。 当前后端是 `poe`。 内部模型键是 `poe_deepseek_v32`。

文档读取 `atasoy_2011_replication/parameter_comparison.csv` 与 `hcm/parameter_comparison.csv`，其中 `gap_ai_minus_human` 定义为 AI 参数减 human 参数。

Atasoy base logit 当前差值最大的参数是 `ASCPMM`，AI 相对 human 更高 `1.4214`。

Exact HCM 当前差值最大的参数是 `beta_ncars`，AI 相对 human 更低 `0.6892`。

## Atasoy Base Logit 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.4134 | 1.0080 | 1.4214 |
| beta_language | 1.0925 | -0.2225 | -1.3151 |
| beta_student | 3.2073 | 2.0322 | -1.1751 |
| beta_work | -0.5824 | 0.2515 | 0.8340 |
| beta_ncars | 1.0010 | 0.2413 | -0.7597 |
| beta_urban | 0.2862 | -0.4476 | -0.7337 |

## Atasoy Base Logit 全部参数

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.4134 | 1.0080 | 1.4214 |
| ASCSM | -0.4700 | 0.2206 | 0.6906 |
| beta_cost | -0.0592 | -0.1130 | -0.0538 |
| beta_distance | -0.2273 | -0.0877 | 0.1396 |
| beta_language | 1.0925 | -0.2225 | -1.3151 |
| beta_nbikes | 0.3469 | 0.1648 | -0.1820 |
| beta_ncars | 1.0010 | 0.2413 | -0.7597 |
| beta_nchildren | 0.1535 | 0.1198 | -0.0337 |
| beta_student | 3.2073 | 2.0322 | -1.1751 |
| beta_time_pmm | -0.0299 | -0.0159 | 0.0140 |
| beta_time_pt | -0.0121 | -0.0201 | -0.0080 |
| beta_urban | 0.2862 | -0.4476 | -0.7337 |
| beta_work | -0.5824 | 0.2515 | 0.8340 |

## Exact HCM 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| beta_ncars | 0.9700 | 0.2808 | -0.6892 |
| alpha_Envir02 | -0.3234 | 0.3613 | 0.6847 |
| alpha_Envir06 | 0.3717 | -0.2790 | -0.6506 |
| beta_work | -0.5830 | -0.0400 | 0.5430 |
| beta_language | 1.0600 | 0.5317 | -0.5283 |
| beta_nbikes | 0.3850 | 0.0802 | -0.3048 |

## Exact HCM 分块参数

### utility

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.5990 | -0.4402 | 0.1588 |
| ASCSM | -0.7720 | -0.8493 | -0.0773 |
| beta_Acar | -0.5740 | -0.7495 | -0.1755 |
| beta_Aenv | 0.3930 | 0.1312 | -0.2618 |
| beta_cost | -0.0559 | -0.1202 | -0.0643 |
| beta_distance | -0.2240 | -0.0951 | 0.1289 |
| beta_language | 1.0600 | 0.5317 | -0.5283 |
| beta_nbikes | 0.3850 | 0.0802 | -0.3048 |
| beta_ncars | 0.9700 | 0.2808 | -0.6892 |
| beta_nchildren | 0.2150 | 0.0173 | -0.1977 |
| beta_student | 3.2600 | 3.1889 | -0.0711 |
| beta_time_pmm | -0.0294 | -0.0165 | 0.0129 |
| beta_time_pt | -0.0119 | -0.0215 | -0.0096 |
| beta_urban | 0.2830 | 0.0447 | -0.2383 |
| beta_work | -0.5830 | -0.0400 | 0.5430 |

### attitude

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| Acar | 3.0200 | 2.8861 | -0.1339 |
| Aenv | 3.2300 | 3.0213 | -0.2087 |
| theta_age | 0.0044 | 0.0088 | 0.0044 |
| theta_basel_zurich | -0.2560 | -0.1415 | 0.1145 |
| theta_bern | -0.3610 | -0.2232 | 0.1378 |
| theta_east | -0.2280 | -0.0088 | 0.2192 |
| theta_educ | 0.2350 | 0.2482 | 0.0132 |
| theta_graubunden | -0.3030 | -0.1456 | 0.1574 |
| theta_nbikes | 0.0845 | 0.0959 | 0.0114 |
| theta_ncars | 0.1040 | 0.1145 | 0.0105 |
| theta_valais | -0.2230 | -0.2627 | -0.0397 |

### measurement

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| alpha_Envir01 | -0.3991 | -0.5280 | -0.1289 |
| alpha_Envir02 | -0.3234 | 0.3613 | 0.6847 |
| alpha_Envir06 | 0.3717 | -0.2790 | -0.6506 |
| alpha_Mobil11 | 0.3937 | 0.4491 | 0.0554 |
| alpha_Mobil16 | 0.3734 | 0.5164 | 0.1430 |
| loading_Envir01 | 0.8381 | 0.9342 | 0.0960 |
| loading_Envir02 | 1.0114 | 0.8496 | -0.1618 |
| loading_Envir06 | 1.0918 | 1.2972 | 0.2054 |
| loading_Mobil11 | 1.1553 | 1.1345 | -0.0207 |
| loading_Mobil16 | 1.0605 | 1.0422 | -0.0183 |
| sigma_Envir01 | 1.3086 | 1.2680 | -0.0407 |
| sigma_Envir02 | 1.1304 | 1.1006 | -0.0299 |
| sigma_Envir05 | 1.0385 | 1.0762 | 0.0376 |
| sigma_Envir06 | 0.7837 | 0.8639 | 0.0802 |
| sigma_Mobil10 | 1.2063 | 1.1043 | -0.1020 |
| sigma_Mobil11 | 1.0977 | 1.1061 | 0.0084 |
| sigma_Mobil16 | 1.1172 | 1.0644 | -0.0528 |
