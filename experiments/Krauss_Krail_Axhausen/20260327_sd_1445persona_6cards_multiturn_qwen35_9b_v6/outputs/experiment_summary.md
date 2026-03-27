# 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- 平均每题调用时长：`0.50` 秒
- 平均每位 respondent 的 6 题总时长：`3.03` 秒
- Sobol draws：`5000`

## 选择分布

- Walking：`4036`
- Bike Sharing：`2243`
- Private Car：`2116`
- E-Scooter Sharing：`275`

## mixed logit 拟合

- final loglikelihood：`-5519.701`
- rho_square：`0.5408`
- 参数数：`36`
- 5% 显著参数数：`23`

## AI 与人类对比

- 可比较参数数：`36`
- 符号一致数：`20`
- sign match rate：`0.5556`

## 5% 显著参数

- `ASC_ES_REL_CAR`：estimate=`6.3953`，se=`1.6312`，z=`3.921`，p=`8.829e-05`
- `ASC_WALK_REL_CAR`：estimate=`2.4175`，se=`1.1876`，z=`2.036`，p=`0.04179`
- `AGE_ES_REL_CAR`：estimate=`-0.1196`，se=`0.0160`，z=`-7.483`，p=`7.283e-14`
- `AGE_BS_REL_CAR`：estimate=`-0.0827`，se=`0.0080`，z=`-10.347`，p=`0`
- `AGE_WALK_REL_CAR`：estimate=`0.1857`，se=`0.0185`，z=`10.044`，p=`0`
- `BIKEACC_BS_REL_CAR`：estimate=`0.2114`，se=`0.0846`，z=`2.498`，p=`0.0125`
- `BIKEACC_WALK_REL_CAR`：estimate=`0.4441`，se=`0.1358`，z=`3.270`，p=`0.001077`
- `CARACC_ES_REL_CAR`：estimate=`-1.3888`，se=`0.3426`，z=`-4.053`，p=`5.047e-05`
- `CARACC_BS_REL_CAR`：estimate=`-1.3923`，se=`0.1229`，z=`-11.325`，p=`0`
- `CARACC_WALK_REL_CAR`：estimate=`-2.0101`，se=`0.1993`，z=`-10.087`，p=`0`
- `PTPASS_ES_REL_CAR`：estimate=`1.2561`，se=`0.5620`，z=`2.235`，p=`0.0254`
- `PTPASS_WALK_REL_CAR`：estimate=`1.6634`，se=`0.3650`，z=`4.557`，p=`5.177e-06`
- `MAAS_ES_REL_CAR`：estimate=`3.0016`，se=`0.8623`，z=`3.481`，p=`0.0004993`
- `MAAS_WALK_REL_CAR`：estimate=`-1.4168`，se=`0.5903`，z=`-2.400`，p=`0.01638`
- `SIGMA_BS`：estimate=`3.1217`，se=`0.1359`，z=`22.973`，p=`0`
- `SIGMA_WALK`：estimate=`-5.8650`，se=`0.3775`，z=`-15.537`，p=`0`
- `B_TIME_BS`：estimate=`1.5011`，se=`0.1192`，z=`12.598`，p=`0`
- `B_TIME_WALK`：estimate=`-0.4445`，se=`0.0353`，z=`-12.594`，p=`0`
- `B_TIME_CAR`：estimate=`1.8217`，se=`0.3319`，z=`5.488`，p=`4.054e-08`
- `B_ACCESS_CAR`：estimate=`1.7651`，se=`0.2353`，z=`7.502`，p=`6.262e-14`
- `B_EGRESS_CAR`：estimate=`-0.2311`，se=`0.0898`，z=`-2.572`，p=`0.0101`
- `B_PARKING`：estimate=`-0.6139`，se=`0.2251`，z=`-2.727`，p=`0.006391`
- `B_COST_LOGMEAN`：estimate=`2.1896`，se=`0.0908`，z=`24.107`，p=`0`
