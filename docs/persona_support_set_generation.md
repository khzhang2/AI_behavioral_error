# 从人类人口统计信息到 persona 支持集与生成流程

本文说明当前代码库中，如何从人类数据的人口统计和出行信息出发，构建 persona 的支持集，并把这些支持集实例化为 AI 问卷实验中的 personas。这里的说明对应当前 intervention-regime 主线代码，核心实现位于 [prepare_optima_intervention_regime_data.py](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/scripts/prepare_optima_intervention_regime_data.py) 和 [optima_intervention_regime_questionnaire.py](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/scripts/optima_intervention_regime_questionnaire.py)。

## 1. 先定义“支持集”是什么意思

这里说的“支持集”，不是把年龄、收入、教育、车辆拥有等单个变量分别列出后，再把它们做笛卡尔积的全枚举集合。当前实现中的支持集是“经验支持集”，也就是只使用人类数据里真实出现过的完整 profile rows。换句话说，代码并不合成一个新的虚拟人口组合空间，而是直接把真实人类样本中的整行 profile 当作 persona 候选。

这种做法的优点是保留了变量之间原本就存在的联合结构。例如，年龄、教育、收入、出行目的、是否有车可用、家庭车辆数和自行车数，在真实数据中往往彼此相关。经验支持集保留了这些相关关系。它的缺点是覆盖能力有限：如果某些 subgroup 在原始人类样本中本来就很少，那么它们在 persona 支持集中也会很少；如果某些组合从未在原始样本中出现，那么当前实现也不会主动生成这些组合。

## 2. 当前 persona 支持集来自哪张表

persona 支持集直接来自 `human_respondent_profiles.csv`。在 prepare 脚本中，这张表被读入后，只保留一组固定字段，形成 `profile_bank`，并写入实验目录中的 `respondent_profile_bank.csv`。

这批字段并不是只有人口统计学变量，而是“人口统计学信息 + 家庭资源 + 出行背景 + 若干行为与态度相关字段”的组合。它至少包括下面几类信息。

| 字段类型 | 当前保留的代表字段 | 用途 |
| --- | --- | --- |
| 基本人口统计信息 | `age`、`sex_text`、`age_text` | 描述 persona 的年龄与性别背景 |
| 收入与教育 | `CalculatedIncome`、`income_text`、`high_education`、`education_text` | 描述社会经济地位 |
| 家庭资源 | `NbCar`、`NbBicy`、`NbHousehold`、`NbChild` | 描述家庭交通资源和家庭规模 |
| 出行上下文 | `work_trip`、`other_trip`、`trip_purpose_text`、`CAR_AVAILABLE`、`car_availability_text` | 描述这次具体出行的背景 |
| 成长与家庭倾向 | `car_oriented_parents`、`childSuburb`、`city_center_as_kid` | 描述成长环境与家庭交通取向 |
| 行为与态度字段 | `Envir01`、`Mobil05`、`LifSty07`、`Envir05`、`Mobil12`、`LifSty01` | 作为 profile bank 的一部分保存，供后续实验整理与对照使用 |

因此，当前 persona 支持集并不是“只有人口统计学信息”的窄支持集，而是“以人口统计和出行背景为主、并保留若干行为态度字段”的完整经验支持集。

## 3. 当前不是正交设计，也不是全枚举

这点需要明确写清。当前代码的 persona 生成规则既不是正交设计，也不是全枚举设计，也不是按人口统计变量做人工平衡的分层设计。当前规则更接近“基于真实 profiles 的可重复随机抽样”。

更具体地说，代码流程是：

1. 从 `human_respondent_profiles.csv` 中取出完整 profile rows。
2. 把这些 rows 作为经验支持集。
3. 对整个支持集做一次可重复的随机打乱。
4. 取前 `M = n_block_templates_per_model` 个 profiles 作为本轮实验使用的 `selected_profiles`。

因此，当前 persona 生成的统计性质更像“从真实支持集中抽完整的人”，而不是“对属性空间做设计”。这意味着：

- 它保留真实联合分布。
- 它不保证 rare groups 一定被覆盖。
- 它不保证每个年龄层、收入层、教育层、车辆拥有层都被平衡抽到。
- 它也不保证跨实验时每一轮都覆盖相同的 subgroup 结构。

## 4. 代码如何从支持集中抽出 personas

在 [prepare_optima_intervention_regime_data.py](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/scripts/prepare_optima_intervention_regime_data.py) 中，persona 相关的核心逻辑可以概括成下面四步。

第一步，读取 `profile_bank`。这一步只是把人类 profile 表裁剪到固定字段集合，不做加权采样，也不做分层。

第二步，打乱 profiles。代码使用：

`profiles.sample(frac=1.0, random_state = master_seed + len(model_key))`

这意味着当前 persona 抽样是可复现的随机打乱。值得注意的是，这里没有使用 `normalized_weight` 来做加权抽样。所以当前抽到哪个 persona，不是按人类样本权重抽的，而是对 profile rows 做等概率置换后再截取。

第三步，处理 `N` 和 `M` 的关系。记：

- `N = profile_bank` 的总行数
- `M = n_block_templates_per_model`

如果 `N >= M`，代码就取打乱后的前 `M` 行。这相当于经验支持集上的无放回随机抽样。

如果 `N < M`，代码会先把整个 `profile_bank` 重复拼接若干次，直到总行数不少于 `M`，再取前 `M` 行。这意味着如果模板数多于原始可用 profile 数，就一定会重复使用某些 persona。

第四步，把 `selected_profiles` 绑定到 block templates。当前代码中，基本上是一份 selected profile 对应一个 `block_template_id`。也就是说，`n_block_templates_per_model` 在当前实现里，几乎等于“本轮实验打算使用多少个 persona-template 单元”。

## 5. persona 真正写进 system prompt 的字段有哪些

虽然 `profile_bank` 保存了很多字段，但真正进入 AI system prompt 的字段只是一部分。当前问卷脚本 [optima_intervention_regime_questionnaire.py](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/scripts/optima_intervention_regime_questionnaire.py) 会把下面这些字段写成 persona 文本。

| 进入 persona 文本的字段 | 在 prompt 中的作用 |
| --- | --- |
| `sex_text`、`age_text` | 描述性别与年龄 |
| `income_text` | 描述收入水平 |
| `trip_purpose_text` | 描述这是 work trip 还是 non-work trip |
| `car_availability_text`、`CAR_AVAILABLE` | 描述这次出行是否有车可用 |
| `NbHousehold`、`NbChild`、`NbCar`、`NbBicy` | 描述家庭规模与交通资源 |
| `education_text` | 描述教育水平 |
| `car_oriented_parents` | 描述家庭成长背景中的交通取向 |
| `childSuburb`、`city_center_as_kid` | 描述童年居住环境 |

因此，当前 persona 的写法不是把 profile 表中所有变量原样灌给模型，而是抽取一组相对可解释、可语言化的人口统计与出行背景字段，把它们写成一个稳定的受访者画像。

## 6. persona 与问卷模板如何绑定

当前实现里，persona 不是单独漂浮存在的。它会和 prompt arm、prompt family、choice card 结构一起，绑定到一个 `block template` 上。

对于每一个被选中的 profile row，代码都会：

1. 生成一个 `block_template_id`。
2. 为这个 template 随机指定一个 `prompt_arm`。
3. 为这个 template 随机指定一个 `prompt_family`。
4. 从 `scenario_bank` 中无放回抽取 `n_core_tasks` 个核心场景。
5. 在这些核心场景上派生 paraphrase、label-mask、order-randomization、monotonicity、dominance 等任务。

这样生成的是“一个 persona 对应的一套问卷模板”。之后再按 `n_repeats_per_template` 把这套模板复制成多次 runs。每次 run 的 `respondent_id` 不同，但它们共享同一 persona、同一模板和同一固定实验条件。

## 7. 从人口统计信息出发，这种方法的统计含义是什么

如果你关心的是“总体平均行为是否像人”，当前方法是有道理的，因为它从真实人类样本中抽出完整 profiles，能够保留真实变量之间的联合关系。

但如果你关心的是“是否覆盖足够广的人群异质性”，当前方法要谨慎解释。原因有三点。

第一，当前抽样不是分层抽样，也不是平衡抽样，所以 rare groups 可能被漏掉。

第二，当前抽样不是按 `normalized_weight` 抽样，而是对支持集等概率打乱后截取。因此，最终被抽中的 persona 结构，未必精确对应人类样本的加权人口结构。

第三，即使 `N` 略大于 `M`，也不能自动保证“覆盖”。如果人口统计异质性是高维的，那么仅仅因为 profile 行数比模板数多，并不意味着所有重要 subgroup 都会被碰到。只有在抽样机制本身做了分层或配额控制时，覆盖性才会更强。

所以当前方法更准确的说法是：它通过经验支持集保留了真实联合结构，但它主要保证“现实性”，不保证“全面覆盖”。

## 8. 如果 `N < M` 会发生什么

当 `N < M` 时，代码会重复 profile rows。这样做可以保证实验仍然能构造出足够多的 `block templates`，但它会带来两个后果。

第一，effective persona sample size 不再等于 `M`，因为一部分 templates 实际上共享同一个基础 profile。

第二，随着重复 profile 变多，实验的横向覆盖能力会下降。此时增加 `n_block_templates_per_model` 不再等价于增加新 persona 的覆盖，而更像是增加同一批 persona 在不同 prompt 或不同 scenario 上的重复暴露。

因此，在解释“persona 异质性覆盖”时，不能只看 `M`，必须同时看 `N` 和抽样机制。

## 9. 当前实现最适合怎样的研究目标

当前 persona 生成方法最适合的目标，是在真实人类 profile 的支持集上，研究 AI 在“看起来像真实受访者”的前提下，是否会表现出随机不稳定性、语义脆弱性、顺序敏感性、trade-off 规则偏差，以及相对 human benchmark 的总体 distortion。

它并不最适合回答“如果我们把人口统计变量做系统正交设计，AI 在所有可能 persona 组合下会怎样”这种问题。后者需要另一套 persona 构造方案，例如分层平衡抽样、约束抽样，或者在经验支持集之上的近似最优设计。

## 10. 当前流程写出的关键文件

如果只关心 persona 支持集与 persona 生成，可以重点看实验目录中的这几个文件。

| 文件 | 含义 |
| --- | --- |
| `respondent_profile_bank.csv` | 从人类 profile 表裁剪出来的经验支持集 |
| `scenario_bank.csv` | 从人类 choice rows 构造出来的场景库 |
| `block_assignments.csv` | persona 与模板、prompt 条件、run repeat 的展开表 |
| `panel_tasks.csv` | 每个 planned run 的全部 task cards |

一句话总结当前实现：当前代码不是从人口统计属性空间中“设计” persona，而是从真实人类 profile 的经验支持集中“抽取” persona，再把它们绑定到 block templates 与 scenarios 上，形成 AI 问卷实验的 respondent blocks。
