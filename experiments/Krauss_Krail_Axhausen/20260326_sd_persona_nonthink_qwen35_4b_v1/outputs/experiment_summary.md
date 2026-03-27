# 实验结果摘要

## 运行摘要

- 模型：`qwen3.5:4b`
- 解码模式：关闭 thinking
- synthetic personas：`30`
- 每个 respondent 的题卡数：`5`
- 总 choice 数：`150`
- valid choice rate：`1.0`
- 平均每个 choice task 的调用时长：约 `4.05` 秒

## 选择分布

- Private Car：`108`
- Walking：`30`
- Bike Sharing：`12`
- E-Scooter Sharing：`0`

## Persona 层面的集中度

- `21` 个 personas 在 5 题里始终选择 `Private Car`
- `6` 个 personas 在 5 题里始终选择 `Walking`
- `2` 个 personas 在 5 题里始终选择 `Bike Sharing`
- `1` 个 persona 在 `Bike Sharing` 和 `Private Car` 之间切换

## AI 与人类参数对比

AI 与 human 的系数对比表在 `ai_vs_human_comparison.csv`。在 16 个可重叠参数的原始符号比较下：

- sign matches：`7`
- sign match rate：`0.4375`

这轮实验中，符号与人类结果大体一致的参数包括：

- `ASC_ES`
- `ASC_WALK`
- `B_TIME_ES`
- `B_ACCESS_SHARED`
- `B_PARKING`
- `B_COST`
- `B_PEDELEC`

这轮实验中，相对于论文出现符号反转的参数包括：

- `ASC_BS`
- `B_TIME_BS`
- `B_TIME_WALK`
- `B_TIME_CAR`
- `B_ACCESS_CAR`
- `B_EGRESS_SHARED`
- `B_EGRESS_CAR`
- `B_AVAILABILITY`
- `B_FREE_FLOATING`

## 解释

这轮 AI replication 已经可以作为第一版 pipeline artifact，但它还不是对人类论文行为结构的高保真恢复。最主要的经验现象是 AI 对 `Private Car` 的高度集中、persona 内切换极少，以及 `E-Scooter` 零选择。这使若干 paper-style 系数的识别很弱，并把部分 robust 标准误推到了极不稳定的状态。

## 主要边界

1. 这是一个 reconstruction-based 的短距离 replication，不是对原始 Ngene blocks 的精确重建。
2. AI 样本量相对于原论文刻意做得很小。
3. 当前 persona prompt 确实诱导出了一些 between-persona heterogeneity，但还不足以干净恢复完整的人类参数结构。
4. 下一轮实验更适合采用更简化的估计规格、更强的 persona 设计，或者加入更多 respondents 与重复题卡。
