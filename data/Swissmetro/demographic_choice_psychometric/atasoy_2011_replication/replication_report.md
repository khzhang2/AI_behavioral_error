# Atasoy, Glerum, and Bierlaire (2011) replication

## Goal

This note reproduces, as closely as possible with the public `optima.dat`, the paper **Attitudes towards mode choice in Switzerland**. The first target is the paper's base logit model. The second target is the paper's continuous hybrid choice model, which combines a mode-choice model with two latent attitudes and continuous measurement equations for selected psychometric indicators.

## Files and command

- Raw data: `/Users/kaihangzhang/Downloads/GitHub/Research codes repo/AI_behavioral_error/data/Swissmetro/demographic_choice_psychometric/raw/optima.dat`
- Replication script: `/Users/kaihangzhang/Downloads/GitHub/Research codes repo/AI_behavioral_error/scripts/replicate_atasoy_2011_models.py`
- Output directory: `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication`
- Re-run command:

```bash
./.venv/bin/python scripts/replicate_atasoy_2011_models.py --output-dir "data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication"
```

## Sample and variable construction

The replication sample keeps all observations with `Choice != -1`. This gives `1906` loop observations from the public Optima file. Choice coding follows the public Optima description: `0 = PT`, `1 = PMM`, `2 = SM`.

For the base logit model, the specification that reproduces Table 7 is the following. Public transport uses `MarginalCostPT`, `TimePT`, `Urban`, and `Student`. Private motorized modes use `CostCarCHF`, `TimeCar`, `NbCar`, `NbChild`, `French`, and `WorkTrip`. Soft modes use `distance_km` and `NbBicy`. The paper-consistent `WorkTrip` dummy is `TripPurpose in {1, 2}`. The paper-consistent household-resource treatment is to recode negative missing codes in `NbCar`, `NbChild`, and `NbBicy` to zero before estimation.

For the continuous model, the same utility-side variables are used, and two latent attitudes are added to the public-transport utility. The pro-car attitude uses the paper's indicators 8, 9, and 10. In the public Optima file these correspond to `Mobil10`, `Mobil11`, and `Mobil16`. The environmental attitude uses the paper's indicators 1, 2, 4, and 5. In the public Optima file these correspond to `Envir01`, `Envir02`, `Envir05`, and `Envir06`. Indicator codes `1` to `5` are treated as valid continuous responses. Codes `6`, `-1`, and `-2` do not contribute to the measurement likelihood.

The structural equations use `NbCar`, `EducHigh`, `NbBicy`, `AgeTerm`, and five region controls. `EducHigh` is defined as `Education >= 6`, which matches the paper's high-education share best. `AgeTerm` is defined as `max(age - 45, 0)`. Region dummies are `Valais = Region 2`, `Bern = Region 4`, `BaselZurich = Region in {5, 6}`, `East = Region 7`, and `Graubunden = Region 8`. The omitted region group is the remaining French-speaking regions.

## Base logit results

The replicated base model log-likelihood is `-1067.356`, compared with the paper's `-1067.4`. This is a near-exact reproduction.

| Parameter | Paper | Ours | Gap |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.4130 | -0.4134 | -0.0004 |
| ASCSM | -0.4700 | -0.4700 | -0.0000 |
| beta_cost | -0.0592 | -0.0592 | 0.0000 |
| beta_time_pmm | -0.0299 | -0.0299 | -0.0000 |
| beta_time_pt | -0.0121 | -0.0121 | 0.0000 |
| beta_distance | -0.2270 | -0.2273 | -0.0003 |
| beta_ncars | 1.0000 | 1.0010 | 0.0010 |
| beta_nchildren | 0.1540 | 0.1535 | -0.0005 |
| beta_language | 1.0900 | 1.0925 | 0.0025 |
| beta_work | -0.5820 | -0.5824 | -0.0004 |
| beta_urban | 0.2860 | 0.2862 | 0.0002 |
| beta_student | 3.2100 | 3.2073 | -0.0027 |
| beta_nbikes | 0.3470 | 0.3469 | -0.0001 |

| Metric | Paper | Ours | Gap |
| --- | ---: | ---: | ---: |
| market share PMM | 0.6231 | 0.6231 | 0.0000 |
| market share PT | 0.3209 | 0.3209 | -0.0000 |
| market share SM | 0.0560 | 0.0560 | -0.0000 |
| elasticity PMM_cost | -0.0640 | -0.0638 | 0.0002 |
| elasticity PMM_time | -0.2470 | -0.2471 | -0.0001 |
| elasticity PT_cost | -0.2160 | -0.2163 | -0.0003 |
| elasticity PT_time | -0.4710 | -0.4710 | -0.0000 |
| value of time PMM (CHF/hour) | 30.30 | 30.35 | 0.05 |
| value of time PT (CHF/hour) | 12.26 | 12.24 | -0.02 |

## Continuous hybrid choice results

The public paper does not report the normalization of the continuous measurement equations. Because of this, the absolute scale of the latent attitudes is not fully identified from the paper text alone. To make the replication explicit and reproducible, the script searches all `3 x 4 = 12` reference-indicator pairs and picks the pair that minimizes the joint gap in Table 7 utility parameters, Table 7 attitude-structure parameters, and the Table 8 choice-only log-likelihood.

The best pair in that deterministic search is `pro-car reference = Mobil10` and `environment reference = Envir05`.

The reported continuous-model coefficients keep the paper-closest local optimum found in that deterministic search-and-refinement procedure. This choice is explicit and reproducible. The paper does not fully disclose the measurement-equation normalization, and later numerical refinements can move to local optima that fit the joint likelihood differently but are clearly less consistent with Tables 7 and 8.

The final continuous model joint log-likelihood is `-18776.464`. Its choice-only log-likelihood is `-1067.490`, compared with the paper's `-1069.8`.

| Utility / attitude parameter | Paper | Ours | Gap |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.5990 | -0.5965 | 0.0025 |
| ASCSM | -0.7720 | -0.7708 | 0.0012 |
| beta_cost | -0.0559 | -0.0577 | -0.0018 |
| beta_time_pmm | -0.0294 | -0.0298 | -0.0004 |
| beta_time_pt | -0.0119 | -0.0121 | -0.0002 |
| beta_distance | -0.2240 | -0.2240 | 0.0000 |
| beta_ncars | 0.9700 | 0.9768 | 0.0068 |
| beta_nchildren | 0.2150 | 0.2019 | -0.0131 |
| beta_language | 1.0600 | 1.0528 | -0.0072 |
| beta_work | -0.5830 | -0.5812 | 0.0018 |
| beta_urban | 0.2830 | 0.2822 | -0.0008 |
| beta_student | 3.2600 | 3.2609 | 0.0009 |
| beta_nbikes | 0.3850 | 0.3747 | -0.0103 |
| beta_Acar | -0.5740 | -0.5662 | 0.0078 |
| beta_Aenv | 0.3930 | 0.3882 | -0.0048 |
| Acar | 3.0200 | 2.9444 | -0.0756 |
| Aenv | 3.2300 | 3.3079 | 0.0779 |
| theta_ncars | 0.1040 | 0.1165 | 0.0125 |
| theta_educ | 0.2350 | 0.2236 | -0.0114 |
| theta_nbikes | 0.0845 | 0.0697 | -0.0148 |
| theta_age | 0.0044 | 0.0036 | -0.0009 |
| theta_valais | -0.2230 | -0.2113 | 0.0117 |
| theta_bern | -0.3610 | -0.3895 | -0.0285 |
| theta_basel_zurich | -0.2560 | -0.2279 | 0.0281 |
| theta_east | -0.2280 | -0.2250 | 0.0030 |
| theta_graubunden | -0.3030 | -0.2838 | 0.0192 |

| Metric | Paper | Ours | Gap |
| --- | ---: | ---: | ---: |
| market share PMM | 0.6311 | 0.6294 | -0.0017 |
| market share PT | 0.3120 | 0.3150 | 0.0030 |
| market share SM | 0.0569 | 0.0556 | -0.0013 |
| elasticity PMM_cost | -0.0580 | -0.0608 | -0.0028 |
| elasticity PMM_time | -0.2340 | -0.2403 | -0.0063 |
| elasticity PT_cost | -0.2020 | -0.2095 | -0.0075 |
| elasticity PT_time | -0.4650 | -0.4713 | -0.0063 |
| value of time PMM (CHF/hour) | 31.54 | 31.02 | -0.52 |
| value of time PT (CHF/hour) | 12.81 | 12.55 | -0.26 |

## Remaining differences

The base logit model is essentially reproduced. The continuous model is closely reproduced on the utility side and on the demand-indicator side, but not exactly on every latent-scale parameter. The main reason is that the paper does not fully document the normalization of the measurement equations. The script makes this ambiguity explicit through the normalization search file and by saving every final measurement parameter.

The paper's Table 12 uses an 80/20 validation split, but the paper does not report the random split seed. For that reason this replication note focuses on Tables 7 to 11, which are the main estimation and demand tables and are directly reproducible from the public data and the public model description.

## External sources used

- Paper PDF: https://transp-or.epfl.ch/documents/technicalReports/AtaGlerBier_2011.pdf
- Optima public description: https://transp-or.epfl.ch/documents/technicalReports/CS_OptimaDescription.pdf
