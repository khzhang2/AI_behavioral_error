# AI Behavioral Error

一个面向学术研究的代码库，用于检验大语言模型生成的"虚拟受访者"能否复现人类在离散选择模型中的行为模式。项目流程：

1. 以 **Optima 数据集**（Swissmetro 出行调查）作为人类基准。
2. 通过 LLM API（Poe、DeepSeek）或本地 **MLX** 小模型收集 AI 问卷回答。
3. 在**干预诊断**、**Atasoy 2011 基础 logit 复现**、**Atasoy 2011 exact HCM** 和 **SALCM** 上比较人类与 AI 的行为差异。

当前 Atasoy 2011 的 AI 分析不再使用一套单独的平行估计代码。`scripts/estimate_atasoy_2011_ai_analysis.py` 会先把 AI 问卷结果整理成与人类复现脚本相同的 Atasoy 行格式输入表，然后复用 `scripts/replicate_atasoy_2011_models.py` 中共享的 base logit 与 exact HCM 模型函数。当前 `data/.../atasoy_2011_replication/hcm/` 下的人类 HCM benchmark 是一个 paper-aligned canonical benchmark：utility 和 attitude 核心参数固定为 paper 表格值，measurement block 则在同一固定 normalization 下补充拟合。AI 侧 exact HCM 仍然在这个 normalization 下用仓库的 local-basin estimator 做数值估计，再与这个 canonical human benchmark 比较。

## 环境配置

### Python 环境

```bash
# macOS / Linux
./.venv/bin/python

# Windows
.\.venv\Scripts\python.exe
```

### 安装依赖

```bash
pip install -e .                    # 核心依赖
pip install -e ".[experiments]"     # + MLX backend 及估计所需的锁定版本
```

版本约束详见 `pyproject.toml`。

### 本地 MLX 小模型

当前仓库唯一的本地小模型 backend 是 `mlx`。默认小量 Qwen 实验使用：

- `active_llm_key = "mlx_qwen35_0p8b"`
- `provider = "mlx"`
- `model = "mlx-community/Qwen3.5-0.8B-5bit"`

这个 backend 直接在 Python 进程内调用 `mlx_lm`，不经过任何本地 HTTP 服务。当前已验证 `mlx_lm 0.31.2` 可以加载这个模型标识。

如果你在 Apple Silicon macOS 上运行本地 MLX 实验，请确保当前环境安装了 `mlx-lm==0.31.2`。当前默认环境就是 `./.venv/bin/python`。

### API 密钥

在项目根目录创建 `api_credentials.local.json`（已加入 gitignore）：

```json
{
  "api_key": "",
  "api_key_env": "POE_API_KEY"
}
```

推荐做法：将 `api_key` 留空，通过环境变量提供密钥：

```bash
# Poe
echo 'export POE_API_KEY="你的密钥"' >> ~/.zshrc && source ~/.zshrc

# DeepSeek
echo 'export DEEPSEEK_API_KEY="你的密钥"' >> ~/.zshrc && source ~/.zshrc
```

**密钥读取顺序：** 模型配置 `api_key` → 凭据文件 → `api_key_env` → 供应商默认环境变量（如 `POE_API_KEY`）。

## 目录结构

```
AI_behavioral_error/
├── data/Swissmetro/demographic_choice_psychometric/
│   ├── raw/                            # 原始 Optima .dat 文件
│   ├── human_cleaned_wide.csv          # 清洗后的人类数据（宽格式）
│   ├── human_respondent_profiles.csv   # 人类受访者画像
│   ├── optima_codebook.json
│   ├── optima_data_description.md
│   └── atasoy_2011_replication/        # 人类基准模型输出
│
├── scripts/                  # 所有可运行脚本（扁平结构，无子包）
├── experiments/Swissmetro/   # 归档实验输出（生成后只读）
│   └── YYYYMMDD_<keywords>_<model>_<version>/
├── docs/                     # 参考文档
│
├── experiment_config.json       # 当前实验覆盖层
├── experiment_config_base.json  # 稳定默认值与方法参数
├── api_credentials.local.json   # 本机 API 密钥（gitignored）
├── pyproject.toml
├── AGENTS.md                    # AI 编程助手指南
└── README.md
```

### 实验文件夹布局

每个实验文件夹只对应**一个模型**。命名规则：`YYYYMMDD_<keywords>_<model>_<version>`。

```
<experiment>/
├── experiment_config.json          # 最终合并后的配置快照
├── outputs/                        # 仅存放原始 AI 采集数据
│   ├── raw_interactions.jsonl
│   ├── respondent_transcripts.json
│   ├── run_respondents.json
│   └── ai_collection_summary.json
├── persona_samples.csv             # 共享派生数据
├── parsed_attitudes.csv
├── parsed_task_responses.csv
├── ai_panel_long.csv
├── ai_panel_block.csv
├── experiment_summary.md
├── atasoy_2011_replication/        # Atasoy 风格整理后输入表 + base logit 结果
├── hcm/                            # exact HCM 结果
└── salcm/                          # SALCM 结果
```

## 配置系统

项目采用**两层配置**：

| 文件 | 角色 |
|---|---|
| `experiment_config.json` | 实验级调参入口，覆盖层 |
| `experiment_config_base.json` | 稳定默认值（估计设置、分析参数） |

`experiment_config.json` 通过 `config_base_file` 引用基线配置，并在 `config_overrides` 中提供覆盖项。运行时两者深度合并，覆盖层优先。

**常用调参字段**（位于 `experiment_config.json`）：

- `experiment_name`、`paths.archive_dir`、`master_seed`
- `active_llm_key`、`llm_models`（每次实验只保留一个条目）
- `n_block_templates_per_model`、`n_repeats_per_template`
- `survey_design.*`（任务数量、prompt 家族/臂、选项顺序）
- `collection.max_workers`

完整字段说明：[`docs/experiment_config_field_reference.md`](docs/experiment_config_field_reference.md)。

## 工作流程

详细步骤指南：[`docs/experiment_workflow.md`](docs/experiment_workflow.md)。

### 快速参考

```bash
# 1. 编辑 experiment_config.json（设置模型、样本量等）

# 2. 构建 persona、场景与问卷模板
./.venv/bin/python scripts/prepare_optima_intervention_regime_data.py --model-key <key>

# 3. 运行 AI 问卷采集
./.venv/bin/python scripts/run_optima_intervention_regime_ai_collection.py \
    --model-key <key> --max-workers 1 [--resume]

# 4. 后续估计
./.venv/bin/python scripts/estimate_optima_intervention_metrics.py
./.venv/bin/python scripts/estimate_atasoy_2011_ai_analysis.py --experiment-dirs <name>
./.venv/bin/python scripts/estimate_optima_salcm.py
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

当前默认后处理会直接复用 `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/` 下的 canonical human benchmark，不会在每次 AI experiment 的 post-AI analysis 时重复重跑 human estimation。

其中 `estimate_atasoy_2011_ai_analysis.py` 这一步会先把 AI 输出重排成与人类复现相同的 Atasoy 输入表，再复用共享的 base logit 与 exact HCM 模型函数。当前 experiment 目录下的 `atasoy_2011_replication/` 与 `hcm/` 子文件夹只保留 AI-side estimate 和 summary，例如 `atasoy_2011_replication/ai_atasoy_base_logit_estimates.csv`、`atasoy_2011_replication/ai_atasoy_base_logit_summary.json`、`hcm/ai_atasoy_hcm_utility_estimates.csv`、`hcm/ai_atasoy_hcm_attitude_estimates.csv`、`hcm/ai_atasoy_hcm_measurement_estimates.csv` 和 `hcm/ai_atasoy_hcm_summary.json`。每个选择模型文件夹还会额外写出一张 `parameter_comparison.csv`，用于并排查看 human 参数和 AI 参数；其中 `salcm/parameter_comparison.csv` 只保留 AI 参数。最后的 `summarize_optima_intervention_regime.py` 会自动读取 `atasoy_2011_replication/parameter_comparison.csv` 与 `hcm/parameter_comparison.csv`，并在 experiment 根目录额外写出一份 `parameter_comparison_report.md`，标题直接使用实验配置里的大模型名称。AI replication input、trace、feasibility 和简短分析说明放回 experiment 根目录。human benchmark 与 paper 对照元数据统一只放在 `data/.../atasoy_2011_replication/`。

只有当 human estimator 或 human specification 本身发生变化时，才应手动刷新 canonical human benchmark，而且刷新结果应继续写回 `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/`，而不是写进某个 experiment 目录：

```bash
./.venv/bin/python scripts/replicate_atasoy_2011_models.py
```

> **注意：** CLI 一律使用 `--model-key`（即配置中的 `key`），而非原始模型名。  
> 若修改了 `survey_design` 或样本量相关字段，需在采集前重新运行 prepare 脚本。

### 采集细节

- **并发粒度**是受访者级别（`--max-workers`），而非单题级别。
- 同一受访者内部串行执行：grounding → attitudes → tasks。
- 当 `provider = "mlx"` 时，collection 脚本会强制把有效并发降到 `1`，避免重复加载模型与显存争抢。
- **`--resume`** 从 `raw_interactions.jsonl` 重建状态，从第一个未完成的受访者/题目继续。
- 每条响应**增量持久化**（JSONL 追加 + CSV 更新），中断最多丢失一次 API 调用。
- 无效 LLM 响应会进行**一次修复尝试**，失败后跳过。

## 脚本

### 当前主线脚本

| 脚本 | 用途 |
|---|---|
| `optima_common.py` | 共享常量、配置加载器、LLM 工具函数 |
| `prepare_optima_data.py` | 清洗原始 Optima 数据 → 人类基准 |
| `prepare_optima_intervention_regime_data.py` | 构建 persona、场景、问卷模板 |
| `optima_intervention_regime_questionnaire.py` | 问卷构造逻辑 |
| `run_optima_intervention_regime_ai_collection.py` | 运行 AI 问卷采集 |
| `estimate_optima_intervention_metrics.py` | 计算干预与随机性诊断指标 |
| `replicate_atasoy_2011_models.py` | 生成人类 Atasoy 2011 base logit 与 paper-aligned exact HCM canonical benchmark |
| `estimate_atasoy_2011_ai_analysis.py` | 将 AI 输出整理成与人类复现一致的 Atasoy 输入表，并复用共享的 base logit 与 exact HCM 模型函数 |
| `write_parameter_comparison_report.py` | 读取 `atasoy_2011_replication/` 与 `hcm/` 的 `parameter_comparison.csv`，在 experiment 根目录生成标题为大模型名称的参数对照 Markdown |
| `estimate_optima_salcm.py` | 尺度调整潜类别模型（SALCM） |
| `summarize_optima_intervention_regime.py` | 生成实验总结报告 |

## LLM 后端配置

每个后端在 `llm_models` 中配置为一个条目。以 Poe 为例：

```json
{
  "key": "poe_gpt54_nano",
  "respondent_prefix": "PO",
  "provider": "poe",
  "model": "gpt-5.4-nano",
  "base_url": "https://api.poe.com/v1",
  "credentials_file": "api_credentials.local.json",
  "format": "json",
  "temperature": 1,
  "seed": 20260412,
  "timeout_sec": 240
}
```

### 各供应商说明

| 供应商 | `base_url` | 备注 |
|---|---|---|
| `poe` | `https://api.poe.com/v1` | 思考开关统一写在 `llm_models[].thinking_mode`；代码会按 `model_behavior_registry.json` 自动映射到底层请求字段 |
| `deepseek` | `https://api.deepseek.com` | 思考开关统一写在 `llm_models[].thinking_mode`；代码会按 `model_behavior_registry.json` 自动映射到底层请求字段 |
| `openai_compatible` | 自定义 | OpenAI-compatible `/chat/completions`；思考开关统一写在 `llm_models[].thinking_mode`；当 `base_url` 指向当前内网 vLLM 服务器时，collection 会自动切到异步 respondent-level 并发分支 |
| `mlx` | 不使用 | 本地模型；`llm_models[].model` 必须是 `mlx_lm.load()` 可直接加载的 Hugging Face MLX repo id，当前默认是 `mlx-community/Qwen3.5-0.8B-5bit` |

### MLX 本地模型约定

- 本地小模型现在只使用 `provider = "mlx"`。
- `llm_models[].model` 对 `mlx` 的含义固定为 `mlx_lm.load()` 可直接加载的模型标识。
- 为了保持 `llm_models` 条目的字段形状一致，`mlx` 也建议显式保留 `base_url`、`credentials_file`、`api_key`、`api_key_env`、`format`、`reasoning_effort`、`top_k`、`timeout_sec`、`extra_body`、`response_decoder` 这些字段，只是把它们写成空占位值。
- 当前仓库对 `mlx` 的建议空值写法是：字符串字段写 `""`，对象字段写 `{}`，数值占位字段如 `top_k`、`timeout_sec` 写 `null`。
- collection 代码会忽略这些 `mlx` 占位字段，不会把它们发到本地 MLX backend。
- `temperature`、`top_p`、`seed`、`grounding_num_predict`、`attitude_num_predict`、`task_num_predict` 会直接映射到本地生成参数。
- 当前默认小量 Qwen 实验键名固定为 `mlx_qwen35_0p8b`。

### 统一思考开关

- 配置里只改 `llm_models[].thinking_mode` 这一处。
- 推荐值只有两个：`"off"` 和 `"on"`。
- 每个模型真正需要的底层写法不直接写进 experiment config，而是记录在仓库根目录的 [model_behavior_registry.json](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/model_behavior_registry.json)。
- 运行时会先按 `model` 自动查这张表，再把 `thinking_mode` 映射成对应的请求字段。例如有的模型用 `reasoning_effort`，有的模型用 `extra_body.chat_template_kwargs.enable_thinking`。
- `reasoning_effort` 现在也视为模型表管理的内部字段。experiment config 里这个键只作为条目形状占位保留，普通实验不要手动改它。
- 如果某个模型没有出现在这张表里，代码就不会替它自动改底层 thinking 参数。

### 自定义响应解码器

可在模型级别覆盖默认的响应解析路径：

```json
{
  "response_decoder": {
    "response_text_path": "choices.0.message.content",
    "thinking_text_path": "choices.0.message.reasoning_content",
    "done_reason_path": "choices.0.finish_reason",
    "prompt_eval_count_path": "usage.prompt_tokens",
    "eval_count_path": "usage.completion_tokens"
  }
}
```

若未提供，代码将根据 `provider` 字段使用默认解析规则。

## 关键约定

- 所有路径基于 `ROOT_DIR = Path(__file__).resolve().parents[1]`（定义在 `optima_common.py`）解析。
- `raw_interactions.jsonl` 和 `respondent_transcripts.json` 已加入 gitignore（大文件）；派生 CSV 会被提交。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
