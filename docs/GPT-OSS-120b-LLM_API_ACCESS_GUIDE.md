# LLM API 访问指南

## 服务端点

| 项目 | 值 |
|------|-----|
| **Base URL** | `http://10.64.89.161:8000/v1` |
| **协议** | OpenAI-compatible (vLLM) |
| **模型名称** | `openai/gpt-oss-120b` |
| **认证** | 无需真实 API Key（任意非空字符串即可） |

该端点运行在内网 A100 GPU 服务器上，通过 [vLLM](https://docs.vllm.ai/) 部署，完全兼容 OpenAI Chat Completions API。

---

## 1. 快速测试

### 1.1 cURL

```bash
curl http://10.64.89.161:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, what model are you?"}
    ],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

### 1.2 查看可用模型

```bash
curl http://10.64.89.161:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

---

## 2. Python 调用

### 2.1 使用 `openai` 库（同步）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://10.64.89.161:8000/v1",
    api_key="EMPTY",          # vLLM 不校验 key，传任意非空值
    timeout=120,
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain travel mode choice in Hong Kong."},
    ],
    temperature=0.0,
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

### 2.2 使用 `openai` 库（异步 / 高并发）

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://10.64.89.161:8000/v1",
    api_key="EMPTY",
    timeout=120,
)

async def batch_predict(prompts: list[str], max_concurrent: int = 15):
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(prompt: str) -> str:
        async with sem:
            resp = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""

    return await asyncio.gather(*[_one(p) for p in prompts])

results = asyncio.run(batch_predict(["Hello", "What is vLLM?"]))
```

### 2.3 使用 `requests` 库（无 SDK 依赖）

```python
import requests

resp = requests.post(
    "http://10.64.89.161:8000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    },
    json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "max_tokens": 256,
    },
    timeout=120,
)
print(resp.json()["choices"][0]["message"]["content"])
```

### 2.4 使用 `litellm`（多 provider 统一接口）

```python
import litellm

response = litellm.completion(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://10.64.89.161:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

---

## 3. 在本项目中使用

项目已经封装了 LLM 客户端，有两种配置方式：

### 3.1 Trip_generator 子项目

直接在 `scripts/run_pipeline.py` 中修改配置变量：

```python
LLM_MODEL       = "openai/gpt-oss-120b"
LLM_PROVIDER    = "openai"
LLM_BASE_URL    = "http://10.64.89.161:8000/v1"
LLM_API_KEY     = None          # None → 自动使用 "EMPTY"
MAX_CONCURRENT  = 15
```

也可在代码中直接构建 `PipelineConfig`：

```python
from src.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    llm_model="openai/gpt-oss-120b",
    llm_provider="openai",
    llm_base_url="http://10.64.89.161:8000/v1",
    llm_api_key=None,
    max_concurrent=15,
    # ... 其他参数
)
run_pipeline(config)
```

### 3.2 trail_project 子项目

编辑 `configs/model/llm_base.yaml`：

```yaml
provider: "A100-localserver"
model: "openai/gpt-oss-120b"
base_url: "http://10.64.89.161:8000/v1"
api_key_env: "A100_LOCAL_API_KEY"
allow_empty_api_key: true
temperature: 0.0
max_tokens: 1024
timeout: 120
use_json_mode: false       # vLLM 不一定支持 response_format=json_object
```

### 3.3 TRAIL_Generator_Project 子项目

编辑 `configs/llm.yaml`：

```yaml
base_url: "http://10.64.89.161:8000/v1"
model: "openai/gpt-oss-120b"
api_key_env: "A100_LOCAL_API_KEY"
allow_empty_api_key: true
max_tokens: 1024
temperature: 0.0
concurrency: 500
```

---

## 4. 可用 API 端点

vLLM 兼容以下 OpenAI API 路由：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | Chat Completions（最常用） |
| `/v1/completions` | POST | Text Completions（旧式） |
| `/v1/embeddings` | POST | 文本嵌入（需模型支持） |

---

## 5. 请求参数参考

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | — | 必须与服务端部署的模型名匹配 |
| `messages` | array | — | Chat 消息列表 |
| `temperature` | float | 0.0 | 控制随机性，0=确定性输出 |
| `max_tokens` | int | 1024 | 最大生成 token 数 |
| `top_p` | float | 1.0 | nucleus sampling |
| `stream` | bool | false | 流式输出 |
| `stop` | string/array | null | 停止词 |
| `n` | int | 1 | 每次生成几个回复 |

---

## 6. 注意事项

1. **网络要求**：该端点位于内网 `10.64.89.161`，需确保客户端在同一网段或有 VPN 访问权限。
2. **无需认证**：vLLM 本地部署不校验 API Key，但 OpenAI SDK 要求非空值，传 `"EMPTY"` 即可。
3. **JSON Mode**：该服务器可能不支持 `response_format={"type": "json_object"}`，建议设置 `use_json_mode: false`，改用 prompt 中的指令引导 JSON 输出。
4. **并发控制**：服务器承载能力有限，建议使用 `asyncio.Semaphore` 控制并发（推荐 15-50）。
5. **超时设置**：推理模型思考时间较长，建议 `timeout >= 120` 秒。
6. **重试策略**：网络不稳定时建议实现指数退避重试（3 次重试 + 2^n 秒等待）。

