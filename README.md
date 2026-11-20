# Kaggle 定时 GPU 任务（路线 A：Qwen3-30B-A3B-GGUF 推理版）

GitHub Actions 当闹钟，Kaggle 2×T4 当算力：到点 push kernel，Kaggle 用 llama.cpp
按层切分跑 GGUF 量化推理，结果落 `/kaggle/working`，webhook 通知。

## 需要的 Secrets（GitHub 仓库 Settings → Secrets → Actions）

- 首选（不过期）：`KAGGLE_USERNAME` + `KAGGLE_KEY`（Kaggle → Settings → API → Create Legacy API Key）
- 可选：`KAGGLE_API_TOKEN`（新版 Token，约 3 小时过期，只适合手动触发）

Workflow 会**优先用 `KAGGLE_API_TOKEN`，未配置时退到 `KAGGLE_USERNAME` + `KAGGLE_KEY`**。

## 使用

1. `kaggle_task/kernel-metadata.json` 里把 `id` / `title` 改成你的用户名和任务名。
2. Kaggle Notebook 侧配置环境变量（Secrets / 环境变量面板）：
   - `WEBHOOK_URL`：企业微信/飞书/钉钉机器人地址（可选，不配则无通知）
   - `GGUF_REPO`：默认 `unsloth/Qwen3-30B-A3B-GGUF`
   - `GGUF_QUANT`：默认 `Q4_K_M`，OOM 改 `Q3_K_L`
   - `N_CTX`：默认 `8192`，OOM 改 `4096`
   - `TASKS_PATH`：任务文件，默认 `/kaggle/input/你的数据集/tasks.jsonl`
3. 准备任务文件 `tasks.jsonl`，每行 `{"id": "...", "prompt": "..."}`，做成私有 Dataset
   挂到 kernel 上。没准备时脚本会跑一条冒烟测试。
4. 推送到 GitHub → Actions 页手动 `Run workflow` 冒烟测试 → 之后按 cron 自动跑。

## 关键约束

- **只推理，不训练**：GGUF 是 llama.cpp 部署格式，不能用 Unsloth/PEFT 微调。
  要微调请换 `unsloth/Qwen3-30B-A3B`（safetensors）另写脚本。
- **必须 T4 ×2**：两张 T4 是各自独立的 16GB，`--split-mode layer --tensor-split 1,1`
  按层均摊；Q4_K_M 约 18.6GB + KV ≈ 20GB，单卡放不下。
- **先选 GPU 再跑**：Session Options → Accelerator 选 `GPU T4 ×2`，且账号要完成手机号验证，
  否则拿不到 GPU 和 Internet 开关。
- **配额**：每周约 30h GPU，按账号算，跑完及时停会话。
- **路径**：`/kaggle/input` 只读；产物写 `/kaggle/working`（约 20GB 上限，会随版本保存）。
- **采样**：思考模式用 `temperature 0.6, top_p 0.95`，别用贪心解码。

## 文件

```
kaggle-schedule/
├── .github/scripts/kaggle.sh          # 凭证 fallback 复用脚本
├── .github/workflows/kaggle-schedule.yml
├── kaggle_task/kernel-metadata.json
├── kaggle_task/main.py                # 路线 A：GGUF 推理 + webhook 通知
├── kaggle_task/requirements.txt       # llama-cpp-python (cu128)
├── .gitignore
└── README.md
```
