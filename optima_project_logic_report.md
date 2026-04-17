# AI Behavioral Error 项目整体逻辑说明

这个项目检验大语言模型作为虚拟受访者时所呈现的行为生成规则。主线流程很清楚：仓库先从 Optima 的 Swissmetro 数据整理出人类基准，再从人类样本里抽取 persona 和场景，生成一套带有系统干预的问卷模板，随后让一个模型在同一 persona 和同一模板上完成多次完整作答，最后用五个 response regime 维度和三类结构模型去定位 AI 与 human 的距离。这里的关键对象有四个。`task` 是一道题，可能是态度题，也可能是 choice card；`block template` 是一整套可复用的问卷模板；`respondent block` 是这套模板在固定实验条件下的一次实例化；`run` 是一轮完整作答，顺序固定为 grounding、attitudes、再加上全部 task cards。当前主线要求一个实验文件夹只对应一个模型，这样每个归档目录都能被直接读成一个完整的实验单元。

## 数据基础与人类基准

人类基准来自 `scripts/prepare_optima_data.py`。这个脚本读取原始 `optima.dat`，保留有效选择观测，写出 `human_cleaned_wide.csv` 和 `human_respondent_profiles.csv` 两张核心表。`human_cleaned_wide.csv` 服务于场景构造和人类结构模型，里面保留了公共交通时间、等待时间、汽车时间、成本、距离以及若干缩放后的版本。`human_respondent_profiles.csv` 服务于 persona 支持集，里面保留了年龄、收入、教育、家庭资源、出行背景、成长环境以及 Atasoy 风格精确 HCM 需要的七个态度指标。

## Persona 如何构建

当前 persona 采用经验支持集。代码直接把 `human_respondent_profiles.csv` 里真实出现过的完整 profile rows 当成候选 persona，并把固定字段写入实验目录中的 `respondent_profile_bank.csv`。这种做法保留了人口统计、家庭资源、出行背景与态度指标之间原本就存在的联合结构，因此 persona 的社会背景和行为背景来自同一条真实人类记录。

抽样规则很直接。`scripts/prepare_optima_intervention_regime_data.py` 先对整个 profile bank 做可复现的等概率打乱，再截取前 `n_block_templates_per_model` 条作为 `selected_profiles`。当模板数大于支持集行数时，脚本会重复拼接 profile bank 直到数量足够，因此模板数表示本轮实验中 persona-template 单元的数量，persona 覆盖范围则由 profile bank 的规模和抽样规则共同决定。

写进 system prompt 的 persona 字段是一组语言上清楚、行为上有解释力的背景变量。当前实现会写入性别、年龄、收入、出行目的、这次出行是否有车可用、家庭人口、子女数、家庭汽车数、自行车数、教育水平、父母是否更依赖汽车，以及童年是否住在 suburb 或 city center。这个设计让模型在整轮问卷里围绕同一个人持续作答，同时把 prompt 控制在足够短、足够稳定的范围内。

## Template 如何构建

template 来自场景支持集和干预规则。`scenario_bank.csv` 由 `human_cleaned_wide.csv` 生成，每一条人类 choice observation 都会被整理成一个三选项场景，并附带 `complexity_score`。这个复杂度分数根据三种方式在简化 proxy 上的最小两两差距来计算，数值越高表示三个选项越接近，任务越难分。这样做让每个 template 同时带着内容和难度信息，后续的 block-level diagnostics 与 SALCM membership covariates 都能直接利用它。

对于每一个被选中的 profile，脚本会先随机指定一个 `prompt_arm` 和一个 `prompt_family`，再从 scenario bank 中无放回抽取 `n_core_tasks` 个核心场景，生成 core tasks。`prompt_arm` 控制模式标签的显式程度，`prompt_family` 控制提示词风格。到这一步，template 已经是一份完整问卷的骨架。

真正让 template 具备 response regime 检验能力的是后续派生出来的 twin 和 probe tasks。paraphrase twin 保持属性不变，只换措辞；label-mask twin 切换语义标签显示方式；order-randomization twin 改变 `A/B/C` 的展示顺序；monotonicity task 把当前 proxy 上最有吸引力的选项变差；dominance task 刻意制造一个在显示属性上明显更差的备选项。每个 template 内部同时包含 core tasks 与这些派生任务，因此同一个 run 既能产出正常选择，也能产出稳定性与规则性检验。

repeat 的作用是 exact repeat。相同 persona、相同 template、相同提示条件会被完整执行多次，因此 repeat 检验的是同一问卷在重复运行中的内部稳定性。twin 和 probe 检验的是同一个底层问题在语言、标签、顺序和规则条件改变后的反应变化。`block_assignments.csv` 记录 run-level 计划表，`panel_tasks.csv` 记录 task-level 计划表，这两张表合起来定义了整轮 AI 实验。

## AI response 实验如何开展

`scripts/run_optima_intervention_regime_ai_collection.py` 负责执行整轮 AI 问卷。每个 `respondent_id` 都会先收到一个 system prompt，再依次完成 grounding、attitude questions 和全部 task cards。grounding 让模型确认 persona 与 trip context，attitude questions 收集七个或更少的指标值，task prompts 要求模型返回结构化 JSON，包括选择标签、信心、两个最重要属性以及是否看到 dominated option。

执行顺序遵循 respondent 内串行、respondent 间并行的原则。串行保证后续题目可以读取之前的回答历史，从而让同一轮问卷更一致；并行提升受访者级吞吐。`mlx` 本地后端会强制使用单 worker，远程后端则使用配置或命令行指定的 `max-workers`。

持久化策略是增量写入。每次 interaction 都会立刻追加到 `outputs/raw_interactions.jsonl`，解析后的 attitude rows 与 task rows 也会立刻写入 CSV，因此中断恢复可以依赖磁盘上已经完成的 respondent 轨迹。collection 完成后，脚本会生成 `persona_samples.csv`、`parsed_attitudes.csv`、`parsed_task_responses.csv`、`ai_panel_long.csv` 和 `ai_panel_block.csv`，其中 long 表服务于结构估计，block 表服务于 regime diagnostics 和 latent class 解释。

## 五个 response regime 维度

### 同一系统的随机不稳定性

第一维测量同一模型在完全相同输入下的内部稳定性。`scripts/estimate_optima_intervention_metrics.py` 会把 exact repeats 按 `block_template_id` 和 `task_index` 聚合，计算 `exact_repeat_flip_rate` 和 `response_entropy`。flip rate 描述两两重复之间的翻转频率，entropy 描述答案分布的离散程度。这个维度给出的是模型自己的噪声基线。

### 语义等价改写是否稳健

第二维测量 semantic invariance。脚本把 anchor task 与 paraphrase twin 的选择分布拿来比较，计算总变差距离 `intervention_gap_tv`，再用 exact repeat 的随机性包络形成 `excess_intervention_gap`。小的 paraphrase gap 表示模型在相同效用结构下保持了稳定的选择规则。正向的 excess intervention gap 表示语义改写带来了超出随机基线的系统偏移。

### 标签或顺序是否过敏

第三维分别测量 label sensitivity 和 order sensitivity。label-mask twin 只切换模式标签的呈现方式，order-randomization twin 只切换 `A/B/C` 的展示顺序，因此两者分别对应语义标签牵引和界面顺序牵引。脚本对这两类 twin 同样计算 flip rate、`intervention_gap_tv` 和 `excess_intervention_gap`。比较时应优先分别阅读 label 和 order 两组指标，因为它们对应两种不同的行为机制。

### 是否真的在做 trade-off

第四维测量 trade-off fidelity。dominance task 检查模型是否会选择一个在显示负担属性上明显更差的选项，核心指标是 `dominance_violation_rate`。monotonicity task 检查模型在某个选项被加重后是否还会向这个更差选项转移，核心指标是 `monotonicity_compliance_rate`。这两个指标一起衡量模型是否真的在读属性并执行一致的权衡规则。

### 相对 human 的结构性失真

第五维测量 human-relative distortion。这里的核心问题是 AI 的稳定规则与人类规则之间有多大距离。基础读法是先看 Atasoy 风格 base logit 的 choice shares 和 share gap，再看系数方向、系数大小、value of time 与 elasticities，最后再把 HCM 与 SALCM 带入解释。这个维度把 AI 放回 human benchmark 的同一结构口径中，因此结论可以直接落到行为机制上。

## MNL、HCM 与 SALCM 分别回答什么

多项 logit 模型，也就是 multinomial logit，简称 MNL，对应仓库里的 Atasoy 风格 base logit。它用三种方式 `PT`、`PMM` 和 `SM` 的效用函数解释选择，并估计方式常数、成本、时间、距离和若干社会经济变量的效应。这个模型最适合读平均行为结构，因为 market shares、elasticities 和 value of time 都很直观。human 与 AI 的第一层比较优先看它。

混合选择模型，也就是 hybrid choice model，简称 HCM，对应当前主线里的 fixed-normalization exact HCM。它在显性属性之外引入两个潜在态度，一个更接近 pro-car attitude，一个更接近 environmental attitude，并用七个 attitude indicators 构造 measurement equations。这样，选择差异可以被继续拆到潜在态度层。这个模型要求七个指标齐全，因此脚本会先写出 `ai_atasoy_hcm_feasibility.json` 再决定是否进入估计。

尺度调整潜类别模型，也就是 scale-adjusted latent class model，简称 SALCM，对应 `scripts/estimate_optima_salcm.py`。它把 AI respondent 群体拆成若干 preference classes 和 scale classes，前者代表偏好结构，后者代表回答尺度或一致性强弱，类别归属再由 `membership_covariates` 解释。这个模型最适合回答 AI 内部是否存在多种 response regimes，以及哪些 regime 更接近 human baseline。

## 估计文件如何解读

读一个实验目录时，建议先看 `outputs/run_respondents.json` 与 `outputs/ai_collection_summary.json`。这两份文件给出完成度、有效率和最基本的 collection 质量。接着看 `exact_repeat_randomness.csv`、`intervention_sensitivity.csv` 和 `intervention_metrics_summary.json`，先建立 AI 自身的稳定性、语义稳健性、标签顺序敏感性与 trade-off fidelity 画像。这个顺序能先把“模型自身如何作答”读清楚。

随后进入 `atasoy_2011_replication/`。`atasoy_replication_input.csv` 是 AI core tasks 被重排成 Atasoy 风格后的估计输入，`base_logit_estimates.csv` 是 MNL 参数，`base_logit_summary.json` 是 market shares、elasticities、value of time 和优化器状态，`base_logit_human_comparison.csv` 直接给出 AI 与 human 的参数差。这里最值得优先比较的是方式常数、时间与成本系数，以及 `PMM`、`PT`、`SM` 的 share gap。

再往后读 `hcm/`。`hcm_utility_estimates.csv` 对应选择效用，`hcm_attitude_estimates.csv` 对应潜在态度结构，`hcm_measurement_estimates.csv` 对应指标测量方程，`hcm_summary.json` 汇总 choice-only log-likelihood、value of time、mean Acar、mean Aenv 和优化器状态，`hcm_human_comparison.csv` 直接给出 AI 与 human 的参数差。优化器状态和参数是否远离初始值决定了 HCM 解释的强度，因此这里应把数值可识别性与行为解释一起读。

最后读 `salcm/`。`ai_salcm_summary.json` 给出收敛状态和各 state 的 posterior mass，`ai_salcm_posterior_membership.csv` 给出 respondent-level 后验归属，`ai_salcm_regime_summaries.csv` 汇总每个 regime 的偏好参数、与 human baseline 的 `normalized_coefficient_distance`、`mode_share_deviation`、以及 label flip、order flip、monotonicity、dominance 等 block-level diagnostics。这个文件最适合回答“AI 内部有哪几类规则”以及“哪一类更接近 human”。

## 外部文献与互联网来源

仓库沿用 `atasoy_2011_replication` 作为内部目录名。外部公开论文对应的是 Bilge Atasoy、Aurélie Glerum、Michel Bierlaire 的 *Attitudes towards mode choice in Switzerland*，发表于 2013 年的 *disP - The Planning Review*，DOI 是 `10.1080/02513625.2013.827518`。DOI 与题名匹配。公开来源可见 [TU Delft Repository 记录](https://repository.tudelft.nl/record/uuid%3A4151b64a-d735-48b4-8ce6-c884c50bcb12) 和 [DOI 链接](https://doi.org/10.1080/02513625.2013.827518)。

关于 stated-preference transport survey 这一实验框架的公开背景，Swiss Federal Office for Spatial Development 提供了 [Access to Stated-Preference-Data](https://www.are.admin.ch/en/access-to-stated-preference-data) 这一类官方说明页面。仓库中的代码逻辑已经把这类调查框架具体化为 persona、template、exact repeat、twin tasks 和结构估计流程，因此阅读实验时应按 `outputs/`、regime diagnostics、`atasoy_2011_replication/`、`hcm/`、`salcm/` 的顺序推进。
