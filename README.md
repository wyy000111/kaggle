# Kaggle 定时 GPU 任务（GitHub Actions 触发）

GitHub Actions 只当闹钟+遥控器，Kaggle 才是真正的算力。

## 目录结构

```
.
├── .github/
│   ├── scripts/
│   │   └── kaggle.sh              # 凭证 fallback 复用脚本（可选）
│   └── workflows/
│       └── kaggle-schedule.yml    # 定时任务
├── kaggle_task/
│   ├── kernel-metadata.json
│   └── main.py
├── .gitignore
└── README.md
```

## 需要的 Secrets（GitHub 仓库 Settings → Secrets → Actions）

二选一，**优先 `KAGGLE_API_TOKEN`**：

- `KAGGLE_API_TOKEN`：Kaggle Settings → API → Generate New Token（`KGAT_` 开头，约 3 小时过期，适合手动/短时触发）
- 或 `KAGGLE_USERNAME` + `KAGGLE_KEY`：Settings → API → Create Legacy API Key（不过期，自动化首选）

两组都配时，workflow 会用 `KAGGLE_API_TOKEN` 并主动 `unset` 另一组，避免错用过期 Token。

可选：`WEBHOOK_URL` 配在 Kaggle 侧 Notebook 的环境变量里（企业微信/飞书/钉钉机器人地址），跑完会推送通知。

## 使用

1. 改 `kaggle_task/kernel-metadata.json` 里的 `id` / `title` 为你的用户名和任务名（两者 slug 必须一致）。
2. 改 `kaggle_task/main.py` 里 `# 2)` 部分为你的真实逻辑。
3. 推送到 GitHub → Actions 页手动 `Run workflow` 冒烟测试 → 之后按 cron 自动跑。

## 改用复用脚本（可选）

把 workflow 里 `run: |` 那一大段换成：

```yaml
- name: Push kernel to Kaggle
  env:
    KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}
    KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
    KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
  run: .github/scripts/kaggle.sh kernels push -p ./kaggle_task
```

记得先 `chmod +x .github/scripts/kaggle.sh` 并提交。

## 注意

- 账号必须先做手机号验证，否则没有 GPU 和联网开关。
- 每周约 30h GPU 配额按账号算，多任务互挤。
- 每次 push 会新建版本并立刻重跑，且会取消同任务正在跑的版本——别设成高频触发。
- `cron` 用 UTC，GitHub 高峰期可能延迟几分钟。
- 产物只写 `./output`；`/kaggle/input` 只读。
- 别把密钥、公司数据写进仓库或 Notebook。
