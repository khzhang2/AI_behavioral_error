# gpt-5.4-nano

这个文档由 `20260415_optima_intervention_regime_poe_gpt54_nano_v1` 下的参数对照表自动生成。 当前后端是 `poe`。 内部模型键是 `poe_gpt54_nano`。

文档读取 `atasoy_2011_replication/parameter_comparison.csv` 与 `hcm/parameter_comparison.csv`，其中 `gap_ai_minus_human` 定义为 AI 参数减 human 参数。

Atasoy base logit 当前差值最大的参数是 `ASCSM`，AI 相对 human 更低 `3.3845`。

Exact HCM 当前差值最大的参数是 `beta_nbikes`，AI 相对 human 更低 `0.9984`。

## Atasoy Base Logit 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCSM | -0.4700 | -3.8545 | -3.3845 |
| beta_student | 3.2073 | 1.2547 | -1.9527 |
| ASCPMM | -0.4134 | 0.5038 | 0.9172 |
| beta_language | 1.0925 | 0.2337 | -0.8588 |
| beta_ncars | 1.0010 | 0.2918 | -0.7092 |
| beta_work | -0.5824 | 0.0644 | 0.6468 |

## Atasoy Base Logit 全部参数

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.4134 | 0.5038 | 0.9172 |
| ASCSM | -0.4700 | -3.8545 | -3.3845 |
| beta_cost | -0.0592 | -0.1044 | -0.0452 |
| beta_distance | -0.2273 | -0.0243 | 0.2030 |
| beta_language | 1.0925 | 0.2337 | -0.8588 |
| beta_nbikes | 0.3469 | -0.0001 | -0.3469 |
| beta_ncars | 1.0010 | 0.2918 | -0.7092 |
| beta_nchildren | 0.1535 | 0.0122 | -0.1414 |
| beta_student | 3.2073 | 1.2547 | -1.9527 |
| beta_time_pmm | -0.0299 | -0.0188 | 0.0111 |
| beta_time_pt | -0.0121 | -0.0198 | -0.0078 |
| beta_urban | 0.2862 | 0.2136 | -0.0726 |
| beta_work | -0.5824 | 0.0644 | 0.6468 |

## Exact HCM 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| beta_nbikes | 0.3850 | -0.6134 | -0.9984 |
| alpha_Mobil16 | 0.3734 | -0.2866 | -0.6600 |
| ASCSM | -0.7720 | -1.2855 | -0.5135 |
| ASCPMM | -0.5990 | -0.0978 | 0.5012 |
| theta_valais | -0.2230 | -0.6524 | -0.4294 |
| alpha_Envir02 | -0.3234 | 0.0878 | 0.4112 |

## Exact HCM 分块参数

### utility

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.5990 | -0.0978 | 0.5012 |
| ASCSM | -0.7720 | -1.2855 | -0.5135 |
| beta_Acar | -0.5740 | -0.4599 | 0.1141 |
| beta_Aenv | 0.3930 | 0.4772 | 0.0842 |
| beta_cost | -0.0559 | -0.1194 | -0.0635 |
| beta_distance | -0.2240 | -0.0399 | 0.1841 |
| beta_language | 1.0600 | 0.9240 | -0.1360 |
| beta_nbikes | 0.3850 | -0.6134 | -0.9984 |
| beta_ncars | 0.9700 | 1.0595 | 0.0895 |
| beta_nchildren | 0.2150 | 0.0066 | -0.2084 |
| beta_student | 3.2600 | 3.2106 | -0.0494 |
| beta_time_pmm | -0.0294 | -0.0172 | 0.0122 |
| beta_time_pt | -0.0119 | -0.0198 | -0.0079 |
| beta_urban | 0.2830 | 0.2420 | -0.0410 |
| beta_work | -0.5830 | -0.2011 | 0.3819 |

### attitude

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| Acar | 3.0200 | 3.3488 | 0.3288 |
| Aenv | 3.2300 | 3.2983 | 0.0683 |
| theta_age | 0.0044 | 0.0095 | 0.0051 |
| theta_basel_zurich | -0.2560 | -0.3113 | -0.0553 |
| theta_bern | -0.3610 | -0.7161 | -0.3551 |
| theta_east | -0.2280 | -0.3327 | -0.1047 |
| theta_educ | 0.2350 | 0.3021 | 0.0671 |
| theta_graubunden | -0.3030 | 0.0490 | 0.3520 |
| theta_nbikes | 0.0845 | 0.1000 | 0.0155 |
| theta_ncars | 0.1040 | -0.0264 | -0.1304 |
| theta_valais | -0.2230 | -0.6524 | -0.4294 |

### measurement

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| alpha_Envir01 | -0.3991 | -0.4740 | -0.0749 |
| alpha_Envir02 | -0.3234 | 0.0878 | 0.4112 |
| alpha_Envir06 | 0.3717 | 0.4951 | 0.1235 |
| alpha_Mobil11 | 0.3937 | 0.4187 | 0.0250 |
| alpha_Mobil16 | 0.3734 | -0.2866 | -0.6600 |
| loading_Envir01 | 0.8381 | 0.8084 | -0.0297 |
| loading_Envir02 | 1.0114 | 0.8363 | -0.1751 |
| loading_Envir06 | 1.0918 | 1.0032 | -0.0887 |
| loading_Mobil11 | 1.1553 | 1.0526 | -0.1026 |
| loading_Mobil16 | 1.0605 | 1.2797 | 0.2192 |
| sigma_Envir01 | 1.3086 | 1.2437 | -0.0650 |
| sigma_Envir02 | 1.1304 | 1.0270 | -0.1034 |
| sigma_Envir05 | 1.0385 | 0.8532 | -0.1854 |
| sigma_Envir06 | 0.7837 | 0.7018 | -0.0819 |
| sigma_Mobil10 | 1.2063 | 1.1462 | -0.0601 |
| sigma_Mobil11 | 1.0977 | 1.0852 | -0.0125 |
| sigma_Mobil16 | 1.1172 | 0.9424 | -0.1748 |
