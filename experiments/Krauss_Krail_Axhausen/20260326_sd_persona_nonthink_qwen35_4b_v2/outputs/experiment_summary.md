# 实验结果摘要

## 运行摘要

- 模型：`qwen3.5:4b`
- 解码模式：`think=false`
- synthetic respondents：`30`
- warm-up 总数：`60`
- 正式 choices：`150`
- valid choice rate：`1.0`
- 全部 `210` 次调用都正常以 `stop` 结束

## 调用时长

- 平均 warm-up 时长：`7.18` 秒
- 平均 choice 时长：`10.03` 秒
- 最长 warm-up 时长：`12.68` 秒
- 最长 choice 时长：`19.48` 秒

## choice 分布

- `E-Scooter Sharing`：`85`
- `Private Car`：`55`
- `Walking`：`10`
- `Bike Sharing`：`0`

## persona 层面的切换

- `28` 个 persona 在 `E-Scooter Sharing` 和 `Private Car` 之间切换
- `2` 个 persona 只选择单一模式

这说明和 v1 相比，这一轮 choice 已经不再被单一模式完全主导，题卡属性对选择的影响更明显。

## mixed logit 结果

- 模型：paper-aligned short-distance panel mixed logit subset
- draw 方法：`5,000 Sobol draws`
- 参数数：`36`
- `final_loglikelihood = -23.419`
- `rho_square = 0.887`

但是当前完整规格在 `5` 张题卡和 `150` 个 choice 上明显过度参数化，很多参数幅度极端，说明识别仍然不稳。例如：

- `ASC_ES_REL_CAR = 10.565`
- `CARACC_ES_REL_CAR = -151.770`
- `PTPASS_ES_REL_CAR = 133.686`
- `B_TIME_CAR = 52.079`
- `B_ACCESS_SHARED = 42.806`
- `B_PARKING = 56.913`

## AI 与人类参数对比

- 可比较参数：`36`
- 符号一致数：`22`
- `sign_match_rate = 0.6111`

这比 v1 有改进，但当前不能把它解释为“已经成功恢复人类真实参数”，因为系数幅度明显失真。

## 当前结论

这轮实验最重要的正面结果是：

- 论文对齐的 mixed logit pipeline 已经用 Python 跑通
- 原文式 `5,000 Sobol draws` 已经复刻
- warm-up + non-thinking 的 respondent 采集稳定
- AI choice 行为比 v1 更受题卡属性驱动

这轮实验最重要的负面结果是：

- `5 cards + 150 choices` 还不足以稳定支撑完整论文规格
- 继续往 prompt 中增加主观行为习惯并不是当前最优方向
- 下一步更应该扩大 design 或收缩估计规格，而不是继续强化 persona
