# 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- 平均每题调用时长：`0.53` 秒
- 平均每位 respondent 的 6 题总时长：`3.21` 秒
- Biogeme draws：`32`

## 选择分布

- Private Car：`5335`
- E-Scooter Sharing：`1340`
- Bike Sharing：`1323`
- Walking：`642`
- Public Transport：`18`
- Car Sharing：`12`

## Biogeme mixed logit 拟合

- final loglikelihood：`-2675.946`
- rho_square：`0.5002`
- 参数数：`67`
- 5% 显著参数数：`35`

## AI 与人类对比

- 可比较参数数：`67`
- 符号一致数：`57`
- sign match rate：`0.8507`

## 5% 显著参数

- `ASC_PT`：estimate=`-13.5543`，se=`6.0619`，z=`-2.236`，p=`0.02535`
- `B_ACCESS_OWNED`：estimate=`-1.0994`，se=`0.3705`，z=`-2.967`，p=`0.003004`
- `B_AGE_BS`：estimate=`-0.0711`，se=`0.0141`，z=`-5.036`，p=`4.759e-07`
- `B_AGE_ES`：estimate=`-0.0878`，se=`0.0162`，z=`-5.404`，p=`6.504e-08`
- `B_AGE_PT`：estimate=`0.0684`，se=`0.0279`，z=`2.454`，p=`0.01412`
- `B_AGE_RP`：estimate=`-0.8083`，se=`0.1864`，z=`-4.335`，p=`1.454e-05`
- `B_AGE_WALK`：estimate=`0.0752`，se=`0.0199`，z=`3.784`，p=`0.0001543`
- `B_AVAILABILITY`：estimate=`0.0402`，se=`0.0036`，z=`11.058`，p=`0`
- `B_BIKEACC_BS`：estimate=`0.7473`，se=`0.1592`，z=`4.694`，p=`2.68e-06`
- `B_BIKEACC_ES`：estimate=`0.5379`，se=`0.1893`，z=`2.842`，p=`0.004485`
- `B_CARACC_BS`：estimate=`-1.4125`，se=`0.2616`，z=`-5.401`，p=`6.645e-08`
- `B_CARACC_ES`：estimate=`-1.3310`，se=`0.2982`，z=`-4.464`，p=`8.05e-06`
- `B_CARACC_PT`：estimate=`-3.5898`，se=`1.4860`，z=`-2.416`，p=`0.0157`
- `B_CARACC_WALK`：estimate=`-1.5128`，se=`0.3418`，z=`-4.425`，p=`9.629e-06`
- `B_MAAS_RP`：estimate=`2.2263`，se=`1.0684`，z=`2.084`，p=`0.03717`
- `B_MAAS_WALK`：estimate=`-3.3913`，se=`0.8859`，z=`-3.828`，p=`0.0001292`
- `B_PEDELEC`：estimate=`-0.5015`，se=`0.2558`，z=`-1.960`，p=`0.04997`
- `B_PTPASS_PT`：estimate=`-1.4255`，se=`0.7124`，z=`-2.001`，p=`0.04539`
- `B_PTPASS_RP`：estimate=`2.4211`，se=`0.9768`，z=`2.479`，p=`0.01319`
- `B_RANGE`：estimate=`0.1430`，se=`0.0332`，z=`4.304`，p=`1.674e-05`
- `B_TIME_BS`：estimate=`-0.2933`，se=`0.0181`，z=`-16.230`，p=`0`
- `B_TIME_CAR`：estimate=`-1.8800`，se=`0.3082`，z=`-6.101`，p=`1.055e-09`
- `B_TIME_ES`：estimate=`-0.2123`，se=`0.0249`，z=`-8.510`，p=`0`
- `B_TIME_PT`：estimate=`-0.4699`，se=`0.0474`，z=`-9.917`，p=`0`
- `B_TIME_RP`：estimate=`-0.8640`，se=`0.0854`，z=`-10.116`，p=`0`
- `B_TIME_WALK`：estimate=`-0.4896`，se=`0.0429`，z=`-11.423`，p=`0`
- `PHI_POOL`：estimate=`0.3459`，se=`0.0378`，z=`9.157`，p=`0`
- `SIGMA_BS`：estimate=`1.2362`，se=`0.2955`，z=`4.184`，p=`2.867e-05`
- `SIGMA_CAR`：estimate=`4.0121`，se=`0.3480`，z=`11.530`，p=`0`
- `SIGMA_COST`：estimate=`-0.4461`，se=`0.0742`，z=`-6.010`，p=`1.854e-09`
- `SIGMA_CS`：estimate=`2.6636`，se=`0.9938`，z=`2.680`，p=`0.007354`
- `SIGMA_ES`：estimate=`2.3837`，se=`0.2766`，z=`8.618`，p=`0`
- `SIGMA_PT`：estimate=`-0.5777`，se=`0.2397`，z=`-2.410`，p=`0.01593`
- `SIGMA_RP`：estimate=`0.8265`，se=`0.2469`，z=`3.348`，p=`0.0008148`
- `SIGMA_WALK`：estimate=`-6.0547`，se=`0.6394`，z=`-9.470`，p=`0`

## 复现实验标签

- `replication_standard = public_materials_high_fidelity`
- `design_provenance = public_exact attribute levels + inferred_from_public block combinations`
- `estimation_backend = biogeme`
