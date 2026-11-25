# Kaggle 定时 GPU 任务（Huihui-Qwen3.6-35B-A3B-GGUF 推理版）

GitHub Actions 当闹钟，Kaggle 2×T4 当算力：到点 push kernel，Kaggle 用挂载的
llama-server 预编译二进制按层切分跑 GGUF 量化推理，通过本地 HTTP API 调用，
结果落 `/kaggle/working`，ntfy 通知。

## 需要的 Secrets（GitHub 仓库 Settings → Secrets → Actions）

- 首选（不过期）：`KAGGLE_USERNAME` + `KAGGLE_KEY`（Kaggle → Settings → API → Create Legacy API Key）
- 可选：`KAGGLE_API_TOKEN`（新版 Token，约 3 小时过期，只适合手动触发）

Workflow 会**优先用 `KAGGLE_API_TOKEN`，未配置时退到 `KAGGLE_USERNAME` + `KAGGLE_KEY`**，
凭证逻辑统一由 `.github/scripts/kaggle.sh` 处理。

## 二进制运行栈（无需编译）

`llama-cpp-bin` 数据集里是**预编译二进制**（`llama-server` + CUDA `.so`），不是 Python wheel。
脚本会把它复制到 `/kaggle/tmp/llama-bin` 并 `chmod +x`，通过 `LD_LIBRARY_PATH` 加载自带 `.so`，
后台启动服务后走 OpenAI 兼容 `/v1/chat/completions`。**不再安装/编译 `llama-cpp-python`**，
省掉每次 ~30min 编译。

- 二进制按 Kaggle T4（CUDA sm_75）编译；换 GPU 架构需换对应二进制。
- 若二进制支持 `--mtp` / `--draft-mtp` 且模型带 MTP head，自动开启投机解码；
  启动失败会逐级回退：带 MTP → 无 MTP → 最小参数集（兼容旧版参数）。
- 日志落 `/kaggle/tmp/llama-server.log`，启动失败时打印尾部 60 行。

## 使用

1. `kaggle_task/kernel-metadata.json` 里 `id` 为 `yongyongwuvip/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF`。
2. Kaggle Notebook 侧配置环境变量（Secrets / 环境变量面板）。**这些参数一律在网页侧设置，GitHub push 不会覆盖也不会写入代码**；换 kernel 或重建时需重新配置：

   | 变量 | 默认值 | 说明 |
   |---|---|---|
   | `TASKS_PATH` | `/kaggle/input/TODO_TASKS_DATASET/tasks.jsonl` | 默认仅为占位路径，**需显式设置**为实际挂载路径，见步骤 3/4；未命中则跑冒烟测试。数据集需挂到 kernel 后才能读到 |
   | `GGUF_CACHE_DIR` | `/kaggle/input/qwen36-35b-a3b-abliterated-mtp-gguf` | 默认对齐 `dataset_sources` 中的 GGUF slug；若 Kaggle 规范化了目录名，需显式覆盖 |
   | `GGUF_REPO` | `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF` | 缓存未命中时的 HF 下载源 |
   | `GGUF_QUANT` | `Q4_K` | 文件名含 `Q4_K`；OOM 改 `Q3_K` |
   | `N_CTX` | `8192` | OOM 改 `4096` |
   | `NGL` | `99` | offload 到 GPU 的层数 |
   | `LLAMA_BIN_DIR` | `/kaggle/input/llama-cpp-bin` | 二进制 Dataset 挂载目录，内含 `llama-server` 与 `.so` |
   | `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `8080` | 本地 HTTP 服务地址 |
   | `SERVER_START_TIMEOUT` | `600` | 等待 `/health` 就绪的秒数 |
   | `REQUEST_TIMEOUT` | `1800` | 单条推理 HTTP 超时（秒） |
   | `ENABLE_MTP` | `1` | 探测到二进制支持时开启 `--mtp --draft-mtp`；设 `0` 强制关闭 |
   | `EXTRA_SERVER_ARGS` | 空 | 追加给 `llama-server` 的参数，空格分隔 |
   | `RUN_LIMIT_HOURS` | `3` | 单次运行上限；到点停止发新请求、落盘已完成结果后退出 |
   | `NTFY_TOPIC` | 空 | ntfy 主题名（留空则不通知；订阅 `https://ntfy.sh/<主题>` 收消息，建议用随机串防误订阅） |
   | `NTFY_SERVER` | `https://ntfy.sh` | 自建 ntfy 服务时改成自己的地址 |
3. 准备任务文件 `tasks.jsonl`，每行 `{"id": "...", "prompt": "..."}`，做成私有 Dataset
   挂到 kernel 上，并在 Notebook 环境变量显式设置
   `TASKS_PATH=/kaggle/input/<数据集slug>/tasks.jsonl`（Kaggle 数据集挂载目录名即 slug，
   不是 dataset title）。没挂载时脚本会跑一条冒烟测试。
4. **数据集挂载**统一写在 `kernel-metadata.json` 的 `dataset_sources` 数组里
   （kernel 每次由 GitHub push，`网页 Add Data` 的临时挂载会被覆盖）：
   ```json
   "dataset_sources": [
     "yongyongwuvip/qwen36-35b-a3b-abliterated-mtp-gguf",
     "yongyongwuvip/llama-cpp-bin"
   ]
   ```
   - `qwen36-35b-a3b-abliterated-mtp-gguf`：GGUF 权重。
   - `llama-cpp-bin`：预编译 llama-server 二进制 + CUDA `.so`，命中日志 `[bin] 使用 llama-server: ...`。
   - 任务数据集准备好后，把其 slug 追加进数组，并同步设置 `TASKS_PATH`。
   - ⚠️ **`dataset_sources` 中每一项必须是真实存在的 slug**：写占位符（如 `TODO_USER/...`）
     会导致 `kaggle kernels push` 返回 **409 Conflict** 在 `SaveKernel`，因为 API 逐项校验数据源。
5. **首跑核对**：日志依次应出现 `[gguf] 命中挂载缓存`、`[bin] 使用 llama-server: ...`、
   `[mtp] 二进制支持 MTP: True/False`、`[server] 就绪 http://127.0.0.1:8080/health`。
   - 若只看到 `下载 ...` 而没有 `[gguf] 命中挂载缓存`，说明 `GGUF_CACHE_DIR` 路径不对，
     用 `GGUF_CACHE_DIR=/kaggle/input/<实际slug>` 覆盖后重跑。
   - 若报 `未找到 llama-server 二进制`，说明 `LLAMA_BIN_DIR` 不对或数据集里没有该可执行文件。
   - 若 `[server] 启动失败` 且日志尾部有参数错误，设 `ENABLE_MTP=0` 或把有问题参数从
     `EXTRA_SERVER_ARGS` 移除；脚本已自带三级回退。
6. 推送到 GitHub → Actions 页手动 `Run workflow` 冒烟测试；同时保留 cron 定时触发（周一至周五 UTC 01:00 / 北京时间 09:00），两种触发方式并存。

### push 报 409 Conflict 的排查

`SaveKernel` 409 是**请求校验失败**（秒级返回），按序检查：

1. `dataset_sources` 每项 slug 是否真实存在且属于凭证账号；占位符/拼错必报 409。
2. `id` 的 owner（`yongyongwuvip`）是否与 KAGGLE_USERNAME 一致；不一致则该 kernel 不属于你。
3. `id` 的 slug 是否与 title 规范化结果一致：Kaggle 会把点号/下划线转连字符、
   大小写归一。用 `kaggle kernels pull -m <owner>/<slug> -p ./meta` 拉回真实元数据比对。
4. `kernel-metadata.json` 里 `is_private` / `enable_gpu` 等字段值是否合法（字符串 `"true"` 可接受）。

## 关键约束

- **只推理，不训练**：GGUF 是 llama.cpp 部署格式，不能用 Unsloth/PEFT 微调。
  要微调请换对应 safetensors 模型（如 `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated`）另写脚本。
- **必须 T4 ×2**：两张 T4 是各自独立的 16GB，`--split-mode layer --tensor-split 1,1`
  按层均摊；Q4_K 约 21.7GB + KV，单卡放不下；OOM 时改 `GGUF_QUANT=Q3_K`（约 17.2GB）。
- **二进制架构**：`llama-cpp-bin` 为 Kaggle T4（sm_75）编译；换 GPU/CUDA 大版本需换二进制。
- **MTP**：需模型含 MTP head 且二进制支持 `--mtp/--draft-mtp`；不支持时自动降级，不影响主推理。
- **先选 GPU 再跑**：Session Options → Accelerator 选 `GPU T4 ×2`，且账号要完成手机号验证，
  否则拿不到 GPU 和 Internet 开关。
- **配额**：每周约 30h GPU，按账号算，跑完及时停会话。
- **路径**：`/kaggle/input` 只读；产物写 `/kaggle/working`（约 20GB 上限，会随版本保存）；
  二进制副本与服务日志写 `/kaggle/tmp`，结束自动清理。
- **采样**：思考模式用 `temperature 0.6, top_p 0.95`，别用贪心解码。

## 文件

```
kaggle-schedule/
├── .github/scripts/kaggle.sh          # 凭证 fallback 复用脚本
├── .github/workflows/kaggle-schedule.yml
├── kaggle_task/kernel-metadata.json
├── kaggle_task/main.py                # 二进制 llama-server + HTTP API + ntfy 通知
├── kaggle_task/requirements.txt       # huggingface_hub（仅 HF 下载兜底用）
├── .gitignore
└── README.md
```
