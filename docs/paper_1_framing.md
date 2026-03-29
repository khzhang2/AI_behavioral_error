# 第一篇文章的 framing

## 暂定标题

当 LLM 作为 travel behavior stated-preference 受访者时，如何识别算法误差与行为误差的组成部分

## 可行性判断

这个 idea 是可行的，但前提是第一篇文章必须被写成一篇“识别与测量”论文，而不是含糊的 “LLM versus human” 比较。最强的版本是：

当 LLM 被用作 stated-preference respondent 时，观测到的 choice variation 至少混合了四类成分：跨 demographic 的差异、同一 demographic 内不同 persona 的差异、prompt 引起的测量差异，以及算法抽样噪声。哪些成分在重复观测下是可识别的？在什么实验设计下可识别？

这个 framing 与方法论导向的 transportation 期刊，比泛泛的 benchmarking 论文更匹配。

## 概念修正

1. 在离散选择中，经典 error term 不只是 “perceived error”。从分析者视角看，它通常吸收所有未观测的效用成分。对这个项目来说，这一点很重要，因为 LLM 的随机性不能直接和稳定偏好异质性混为一谈。
2. LLM 中的 persona heterogeneity 并不天然等价于人类受访者异质性。persona 往往是由 prompt 诱导出来的，因此它部分属于测量工具，而不只是 respondent 本身。
3. Prompt variation 不是事后的 nuisance，它本身就是一种 survey-design 干预，会改变被测得的 choice rule。
4. Seed、temperature 和 provider 设置并不能完全定义模型的随机状态。还必须记录 model version、backend 实现和 API 日期，因为跨版本后 determinism 可能会失效。
5. 如果没有在同一个 task-persona-prompt 单元内做重复观测，就无法把单元内随机性和 latent instability 区分开。

## 最小数学设定

令 respondent 单元 `(d, p, q, r)` 分别表示 demographic scaffold `d`、persona instantiation `p`、prompt version `q` 和重复抽样 `r`。对于任务 `t` 和备选项 `j`，写成：

`U_dpqtrj = V(x_tj ; beta_d + eta_dp + pi_q) + tau_tj + epsilon_dpqtrj`

其中：

- `beta_d` 表示跨 demographic 的偏好结构。
- `eta_dp` 表示同一 demographic 内不同 persona 的偏离。
- `pi_q` 表示 prompt 引起的决策规则偏移。
- `tau_tj` 表示未被其他参数化项显式表示的题卡特征。
- `epsilon_dpqtrj` 表示单元内残余噪声。

观测到的选择为：

`y_dpqtr = argmax_j U_dpqtrj`

在 LLM 实验里，`epsilon_dpqtrj` 不是一个单一原语。一个有用的分解方式是：

`epsilon_dpqtrj = a_rj + m_dpqtrj`

其中 `a_rj` 表示由 decoding 与 backend 随机性引起的算法抽样波动，`m_dpqtrj` 表示未被 `pi_q` 吸收的残余测量不稳定性，例如 wording sensitivity。

## 识别命题

### 命题 1

如果每个 `(d, p, q, t)` 单元只有一个观测，那么稳定的单元内偏好差异与算法随机性无法仅凭 choice outcome 分开识别。

### 命题 2

在固定 model version、题卡、prompt 和 persona，只改变 decoding 随机性的重复调用下，可以识别算法方差的一个下界。

### 命题 3

对 demographics、personas、prompts 和 repeated calls 做 crossed design 时，只有每个因素内部都有 replication，才能进行方差分解。否则相应的方差成分会与残差噪声混叠。

### 命题 4

识别质量依赖于任务信息量。当确定性效用差距很小的时候，随机 choice reversal 会在观测上与 taste heterogeneity 混淆。

### 命题 5

稳健的参数恢复不仅要报告点估计，还必须报告 prompt versions、decoding settings 和 alternative ordering 变化下的 sensitivity envelope。

## 建议的论文结构

1. 问题提出：LLM 正越来越多地被当作 synthetic respondents 使用，但观测到的 response variation 混合了稳定偏好结构与算法随机性。
2. 形式化框架：定义 choice model 和方差分解目标。
3. 识别分析：说明在不同 replication 方案下，哪些成分能识别、哪些不能识别。
4. 实验设计：先以当前保留下来的 Optima 数据集做基线识别与 smoke test，再决定是否扩展到新的 stated-preference reconstruction 设计。
5. 实证结果：解析稳定性、顺序效应、基线 MNL 恢复，以及基于重复运行的方差分解。
6. Trustworthiness 讨论：稳健性、验证、审计轨迹和不确定性边界。

## 执行顺序

1. 将第一篇文章固定在 short-distance。
2. 明确把当前问卷定义为 reconstruction-based pilot，而不是原始 Ngene 设计的精确复刻。
3. 先运行 neutral-prompt baseline。
4. 保留 respondent panel 结构：一个 synthetic respondent 要连续完成全部题卡。
5. 先通过 smoke test，再扩样本。
6. 只有基线跑通后，才进入 repeated-seed、repeated-prompt 和 repeated-persona 的误差分解实验。
