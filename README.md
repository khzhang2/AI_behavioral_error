# AI Behavioral Error

一个面向学术研究的代码库，用于检验大语言模型生成的"虚拟受访者"能否复现人类在离散选择模型中的行为模式。项目流程：

1. 以 **Optima 数据集**（Swissmetro 出行调查）作为人类基准。
2. 通过 LLM API（Poe、DeepSeek）或本地 **MLX** 小模型收集 AI 问卷回答。
3. 在**干预诊断**、**Atasoy 2011 基础 logit 复现**、**Atasoy 2011 exact HCM** 和 **SALCM** 上比较人类与 AI 的行为差异。

当前 Atasoy 2011 的 AI 分析不再使用一套单独的平行估计代码。`scripts/estimate_atasoy_2011_ai_analysis.py` 会先把 AI 问卷结果整理成与人类复现脚本相同的 Atasoy 行格式输入表，然后直接调用 `scripts/replicate_atasoy_2011_models.py` 中同一套 base logit 与 exact HCM 估计函数。这样，AI 结果与人类结果可以在同一口径下直接比较。

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
pip install -e ".[experiments]"     # + Biogeme、MLX backend 及估计所需的锁定版本
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
│   ├── shared_sobol_draws_{32,500}.npy # 预生成的 Sobol 抽样
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
├── biogeme.toml                 # Biogeme 默认参数（gitignored）
├── biogeme_runtime.toml         # 由脚本写入（gitignored）
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
./.venv/bin/python scripts/replicate_atasoy_2011_models.py
./.venv/bin/python scripts/estimate_atasoy_2011_ai_analysis.py --experiment-dirs <name>
./.venv/bin/python scripts/estimate_optima_salcm.py
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

其中 `estimate_atasoy_2011_ai_analysis.py` 这一步会先把 AI 输出重排成与人类复现相同的 Atasoy 输入表，再复用人类复现脚本里的 base logit 与 exact HCM 估计函数。当前实验归档因此会同时保留人类风格的输出文件，例如 `atasoy_2011_replication/base_logit_estimates.csv`、`atasoy_2011_replication/base_logit_summary.json`、`hcm/hcm_utility_estimates.csv`、`hcm/hcm_attitude_estimates.csv`、`hcm/hcm_measurement_estimates.csv` 和 `hcm/hcm_summary.json`；原来的 `ai_atasoy_*` 文件名仍会继续写出，用来兼容旧下游脚本。

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
| `replicate_atasoy_2011_models.py` | 复现人类 Atasoy 2011 base logit 与 exact HCM |
| `estimate_atasoy_2011_ai_analysis.py` | 将 AI 输出整理成与人类复现一致的 Atasoy 输入表，并复用同一套 base logit 与 exact HCM 估计代码 |
| `estimate_optima_salcm.py` | 尺度调整潜类别模型（SALCM） |
| `summarize_optima_intervention_regime.py` | 生成实验总结报告 |

### 旧版脚本（仅供参考）

`legacy_estimate_optima_biogeme_hcm.py`、`legacy_estimate_optima_torch_hcm.py`、`legacy_compare_optima_hcm.py`、`legacy_optima_hcm_model_spec.py` — 不属于当前实验流水线。

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
| `poe` | `https://api.poe.com/v1` | 设 `reasoning_effort` 为 `"none"` 可关闭推理模式 |
| `deepseek` | `https://api.deepseek.com` | 使用 `thinking_mode`：`"non_thinking"` 或 `"thinking"` |
| `mlx` | 不使用 | 本地模型；`llm_models[].model` 必须是 `mlx_lm.load()` 可直接加载的 Hugging Face MLX repo id，当前默认是 `mlx-community/Qwen3.5-0.8B-5bit` |

### MLX 本地模型约定

- 本地小模型现在只使用 `provider = "mlx"`。
- `llm_models[].model` 对 `mlx` 的含义固定为 `mlx_lm.load()` 可直接加载的模型标识。
- 为了保持 `llm_models` 条目的字段形状一致，`mlx` 也建议显式保留 `base_url`、`credentials_file`、`api_key`、`api_key_env`、`format`、`reasoning_effort`、`top_k`、`timeout_sec`、`extra_body`、`response_decoder` 这些字段，只是把它们写成空占位值。
- 当前仓库对 `mlx` 的建议空值写法是：字符串字段写 `""`，对象字段写 `{}`，数值占位字段如 `top_k`、`timeout_sec` 写 `null`。
- collection 代码会忽略这些 `mlx` 占位字段，不会把它们发到本地 MLX backend。
- `temperature`、`top_p`、`seed`、`grounding_num_predict`、`attitude_num_predict`、`task_num_predict` 会直接映射到本地生成参数。
- 当前默认小量 Qwen 实验键名固定为 `mlx_qwen35_0p8b`。

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
- Sobol 抽样**预先生成并共享**给所有估计方法，以确保可复现性。
- Biogeme 的 `*.iter`、`biogeme.toml`、`biogeme_runtime.toml` 已加入 gitignore；运行时 TOML 在每次估计时重新写入。
- `raw_interactions.jsonl` 和 `respondent_transcripts.json` 已加入 gitignore（大文件）；派生 CSV 会被提交。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
