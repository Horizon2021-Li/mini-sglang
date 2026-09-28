# Mini-SGLang 安装与源码阅读

## 参考与版本

- [官方 Quick Start](https://github.com/sgl-project/mini-sglang#-quick-start)
- 用户提供的 [Code Structure](https://mintlify.wiki/sgl-project/mini-sglang/api/code-structure) 本次无法访问；改为阅读仓库官方 [架构文档](docs/structures.md)，并对照实际源码。
- 源码版本：`9a91cfafe754aa85daee49998176275667eb58f2`。
- 项目路径：`/home/horizon/Code/ai_infra/mini-sglang`。
- 独立环境：项目内 `.venv`，Python 3.12。

## 模块职责

| 模块 | 职责 | 建议入口 |
| --- | --- | --- |
| `core` | 共享的数据结构和进程内推理上下文 | [core.py](python/minisgl/core.py) |
| `scheduler` | 接收请求、组批，安排 prefill/decode，协调缓存资源与 Engine | [scheduler.py](python/minisgl/scheduler/scheduler.py) |
| `engine` | 管理单个 TP worker 的模型、上下文、KV pool、注意力后端、CUDA Graph 和采样 | [engine.py](python/minisgl/engine/engine.py) |
| `layers` | 线性层、embedding、RMSNorm、RoPE、Attention 等可复用组件，支持张量并行 | [attention.py](python/minisgl/layers/attention.py) |
| `models` | 将 layers 组合成 Llama、Qwen 等模型；处理模型配置、权重加载与分片 | [qwen3.py](python/minisgl/models/qwen3.py)、[weight.py](python/minisgl/models/weight.py) |
| `attention` | 统一注意力后端接口，准备批次元数据，调用 FlashInfer、FlashAttention 或 TRT-LLM 实现 | [base.py](python/minisgl/attention/base.py) |
| `kvcache` | 存储各层 K/V 张量；维护前缀匹配、插入、锁定与淘汰 | [base.py](python/minisgl/kvcache/base.py)、[radix_cache.py](python/minisgl/kvcache/radix_cache.py) |
| `distributed` | TP rank 信息与 all-reduce/all-gather 等集合通信 | [distributed](python/minisgl/distributed) |
| `server` / `tokenizer` / `message` | HTTP 接口、进程启动、文本与 token 转换、ZMQ 消息协议 | [launch.py](python/minisgl/server/launch.py) |
| `kernel` | 项目自定义 C++/CUDA 算子及 TVM-FFI JIT 编译封装 | [utils.py](python/minisgl/kernel/utils.py) |
| `llm` | 面向 Python 使用者的推理入口 | [llm.py](python/minisgl/llm/llm.py) |

### core：Req / Batch / Context

- `Req` 表示单条请求。`input_ids` 是 CPU token 张量；`cached_len` 表示已缓存长度；`device_len` 跟踪设备侧序列进度；`table_idx` 关联页表；`cache_handle` 引用前缀缓存。`extend_len = device_len - cached_len` 是本轮还需处理的 token 数。重叠调度中，设备进度和 CPU token 更新是分开的。
- `Batch` 表示一次执行的请求集合，`phase` 区分 prefill/decode。调度器填写 `input_ids`、`positions`、`out_loc` 和 padding 信息；注意力后端填写 `attn_metadata`。
- `Context` 持有当前进程的页表、KV pool、注意力/MoE 后端及当前 batch。`forward_batch()` 临时设置活动 batch，退出时清理；不允许嵌套。它是进程内共享状态，多 GPU 的各个 worker 各有自己的实例。
- `SamplingParams` 保存温度、top-k、top-p、输出长度和 EOS 策略。

### 两个容易混淆的边界

1. `scheduler` 决定本轮算谁、算多少；`engine` 负责执行这批计算。
2. KV pool 保存实际 GPU K/V 张量；前缀缓存保存 token 前缀到缓存位置的映射及复用/淘汰状态。`scheduler/cache.py` 再协调空闲页和请求资源分配。仓库架构文档中的部分 CacheManager 命名与当前 `BasePrefixCache` 接口不同，应以源码为准。

### 请求的数据流

```text
用户 → API Server → Tokenizer → Scheduler → Engine → Models → Layers
                                  ↑                    ↓       ↓
                          更新请求、继续 decode       Context ↔ Attention / KVCache
                                  ↓
用户 ← API Server ← Detokenizer ← 输出 token
```

控制消息使用 ZMQ；多 GPU 张量并行使用 NCCL 等集合通信。Prefill 处理提示词中未缓存的部分并写入 KV；decode 逐步使用历史 KV 生成后续 token。共享前缀可通过 radix cache 减少重复 prefill。

## 阅读顺序

1. `core.py`：弄清 Req、Batch、Context 的字段和生命周期。
2. `scheduler/scheduler.py`，配合 `prefill.py`、`decode.py`、`cache.py`：跟踪组批与资源管理。
3. `engine/engine.py`，配合 `graph.py`、`sample.py`：跟踪一次前向和采样。
4. `models/qwen3.py` → `layers/attention.py` → `attention/fi.py`：从模型逐层追到后端。
5. `kvcache/mha_pool.py` 与 `radix_cache.py`：区分真实数据存储和前缀索引。
6. `server/launch.py` 与 `message/`：把单进程计算放回完整服务的数据流中。


## 使用环境与启动

在 Bash 中执行：

```bash
cd /home/horizon/Code/ai_infra/mini-sglang
source ./activate-minisgl.sh
python -m minisgl --help
```

本机是 WSL/Linux，GPU 为 RTX 5080 16 GB（compute capability 12.0）。安装了 PyTorch `2.9.1+cu128`、Transformers `4.57.3`、FlashInfer `0.7.0`、sgl-kernel `0.3.21`。完整 Python 依赖版本见 [minisgl-installed.txt](minisgl-installed.txt)。

原环境没有 `nvcc` 和 `libnuma.so.1`。已从 NVIDIA 官方 redistributable 获取 `cuda_nvcc`、`cuda_cudart`、`cuda_cccl`、`libcurand`，校验 SHA-256 后安装到本地。最初安装 CUDA 12.8，后因当前 FlashInfer 对 SM 12.x 要求 CUDA >=12.9，改用 `.cuda-12.9`。这些是编译所需组件，不是包含全部工具的系统级 Toolkit。libnuma1 从 Ubuntu 软件源下载并解压到 `.system-libs`。激活脚本配置以上路径。

`nvidia-smi` 显示的是驱动支持的 CUDA 能力；PyTorch 的构建版本、JIT 使用的 Toolkit 版本应分别检查。

### 启动模型（首次运行下载权重）

```bash
source /home/horizon/Code/ai_infra/mini-sglang/activate-minisgl.sh
python -m minisgl --model Qwen/Qwen3-0.6B --attention-backend fi \
  --memory-ratio 0.65 --max-seq-len-override 4096 \
  --max-running-requests 8 --cuda-graph-max-bs 8
```

这是为桌面 GPU 预留显存的起始配置，尚未做完整模型推理验证。服务默认地址是 `http://127.0.0.1:1919`。终端交互可添加 `--shell-mode`；Hugging Face 下载不可用时可添加 `--model-source modelscope`。

服务就绪后可请求：

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"你好，请用一句话介绍你自己。"}],"max_tokens":128}'
```

### 已验证

- `uv pip check`：93 个包依赖检查通过。
- RTX 5080 上 PyTorch CUDA 张量求和结果正确。
- `python -m minisgl --help` 成功。
- FlashInfer 与 sgl-kernel 导入成功。
- Mini-SGLang 自定义 CUDA KV 写入内核 JIT 编译及执行通过：写入两个非连续槽位，K/V 结果与输入逐元素一致。
- CUDA 组件提供 `lib`，已添加 `lib64 -> lib` 兼容链接供 TVM-FFI 链接器使用。
- 安装阶段未启动持久服务；后续 Llama 权重加载与 Req/Batch 构造见 [动手任务](exercises/llama-checkin/README.md)，端到端模型推理尚未验证。


详细源码学习见 [models / layers / core 阅读笔记](mini-sglang-source-study.md)。
