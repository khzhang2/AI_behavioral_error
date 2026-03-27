# 实验结果摘要

## 运行摘要

- synthetic respondents：`100`
- choice tasks per respondent：`6`
- 总 choices：`600`
- valid choice rate：`1.0000`
- 平均每题调用时长：`0.42` 秒
- 平均每位 respondent 的 6 题总时长：`2.53` 秒
- Sobol draws：`5000`

## 选择分布

- Private Car：`440`
- Bike Sharing：`115`
- Walking：`26`
- E-Scooter Sharing：`19`

## mixed logit 拟合

- final loglikelihood：`-325.800`
- rho_square：`0.6083`
- 参数数：`36`
- 5% 显著参数数：`7`

## AI 与人类对比

- 可比较参数数：`36`
- 符号一致数：`26`
- sign match rate：`0.7222`

## 5% 显著参数

- `AGE_ES_REL_CAR`：estimate=`-0.0500`，se=`0.0199`，z=`-2.518`，p=`0.01179`
- `BIKEACC_ES_REL_CAR`：estimate=`-0.7097`，se=`0.2588`，z=`-2.743`，p=`0.006095`
- `CARACC_BS_REL_CAR`：estimate=`-1.2545`，se=`0.2433`，z=`-5.157`，p=`2.513e-07`
- `PTPASS_ES_REL_CAR`：estimate=`1.1470`，se=`0.5543`，z=`2.069`，p=`0.03854`
- `SIGMA_BS`：estimate=`1.2785`，se=`0.2697`，z=`4.740`，p=`2.138e-06`
- `B_COST_LOGMEAN`：estimate=`-4.6226`，se=`2.3463`，z=`-1.970`，p=`0.04882`
- `SIGMA_COST`：estimate=`-3.0407`，se=`0.9721`，z=`-3.128`，p=`0.00176`
