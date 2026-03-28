# Torch 实验结果摘要

## 运行摘要

- synthetic respondents：`10`
- choice tasks per respondent：`6`
- 总 choices：`60`
- valid choice rate：`1.0000`
- draws：`32`
- device：`cuda`
- runtime_seconds：`7.76`

## 选择分布

- Private Car：`5335`
- E-Scooter Sharing：`1340`
- Bike Sharing：`1323`
- Walking：`642`
- Public Transport：`18`
- Car Sharing：`12`

## Torch mixed logit 拟合

- final loglikelihood：`-6.464`
- init loglikelihood：`-59.201`
- null loglikelihood：`-83.178`
- rho_square_vs_init：`0.8908`
- rho_square_vs_null：`0.9223`

## Torch vs Biogeme(32 draws)

- 可比较参数数：`0`
- 符号一致数：`0`
- sign match rate：`0.0000`

## Torch vs Human

- 可比较参数数：`67`
- 符号一致数：`42`
- sign match rate：`0.6269`
