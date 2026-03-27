# 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- 平均每个 session 时长：`8.87` 秒
- Sobol draws：`5000`

## 选择分布

- Bike Sharing：`3547`
- Private Car：`3535`
- E-Scooter Sharing：`1588`

## mixed logit 拟合

- final loglikelihood：`-3379.773`
- rho_square：`0.7188`
- 参数数：`36`
- 5% 显著参数数：`6`

## AI 与人类对比

- 可比较参数数：`36`
- 符号一致数：`25`
- sign match rate：`0.6944`

## 5% 显著参数

- `PTPASS_BS_REL_CAR`：estimate=`-15.7634`，se=`2.2083`，z=`-7.138`，p=`9.459e-13`
- `B_TIME_BS`：estimate=`-8.4816`，se=`2.1403`，z=`-3.963`，p=`7.41e-05`
- `B_ACCESS_SHARED`：estimate=`-19.4383`，se=`2.8434`，z=`-6.836`，p=`8.13e-12`
- `B_ACCESS_CAR`：estimate=`-40.6473`，se=`5.7573`，z=`-7.060`，p=`1.664e-12`
- `B_EGRESS_SHARED`：estimate=`-247.8140`，se=`13.0588`，z=`-18.977`，p=`0`
- `SIGMA_COST`：estimate=`-2.0712`，se=`0.0289`，z=`-71.636`，p=`0`
