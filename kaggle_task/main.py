"""Kaggle 2×T4 定时推理：Qwen3-30B-A3B-GGUF（llama.cpp 按层切分）。

用途：GitHub Actions 定时 push 到 Kaggle，Kaggle 侧用 2 张 T4 跑批量推理。
说明：
  - 只做推理，不做训练。GGUF 是 llama.cpp 部署格式，不能用 Unsloth/PEFT 微调；
    要微调请换成 unsloth/Qwen3-30B-A3B（safetensors）另写脚本。
  - 两张 T4 是各自独立的 16GB，必须 --split-mode layer --tensor-split 1,1 按层均摊。
  - Q4_K_M 文件约 18.6GB + KV ≈ 20GB，2×T4 合计 32GB 能塞下；
    若 OOM 请把 GGUF_REPO 换成 Q3_K_L 并把 N_CTX 调到 4096。
  - 产物一律写 /kaggle/working（会随版本保存），别写 /kaggle/input（只读）。
"""
import os
import json
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

# ---------------- 配置（可用 Kaggle Notebook 环境变量覆盖） ----------------
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/Qwen3-30B-A3B-GGUF")
GGUF_QUANT = os.environ.get("GGUF_QUANT", "Q4_K_M")   # OOM 时改 Q3_K_L
N_CTX = int(os.environ.get("N_CTX", "8192"))          # OOM 时降到 4096
NGL = int(os.environ.get("NGL", "99"))                # 全部层 offload 到 GPU
PORT = int(os.environ.get("PORT", "8080"))
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
TASKS_PATH = os.environ.get("TASKS_PATH", "/kaggle/input/你的数据集/tasks.jsonl")
OUT_DIR = "/kaggle/working/out"

# 思考模式官方推荐采样参数；不要用贪心解码
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 2048}


def notify(text: str) -> None:
    """best-effort 通知，失败不影响主流程。"""
    if not WEBHOOK_URL:
        return
    payload = json.dumps({"msgtype": "text", "text": {"content": text}}).encode()
    req = urllib.request.Request(
        WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def install_stack() -> None:
    """装 llama.cpp Python 绑定（CUDA 版）+ 依赖。"""
    log("安装 llama-cpp-python (cu128) ...")
    subprocess.run(
        "pip install llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128",
        shell=True, check=True,
    )


def locate_gguf() -> str:
    """下载 GGUF 并返回本地 .gguf 文件路径。"""
    log(f"下载 {GGUF_REPO}:{GGUF_QUANT} ...")
    subprocess.run(
        f"huggingface-cli download {GGUF_REPO} "
        f"--include '*{GGUF_QUANT}*' --local-dir /kaggle/tmp/gguf --local-dir-use-symlinks False",
        shell=True, check=True,
    )
    for root, _dirs, files in os.walk("/kaggle/tmp/gguf"):
        for f in files:
            if f.endswith(".gguf"):
                return os.path.join(root, f)
    raise RuntimeError("未找到 .gguf 文件，检查 GGUF_QUANT 拼写")


def start_server(gguf_path: str) -> subprocess.Popen:
    """起 llama-server：按层切到 2×T4，Flash Attention + 8bit KV 省显存。"""
    cmd = (
        f"llama-server -m {gguf_path} -ngl {NGL} "
        f"--split-mode layer --tensor-split 1,1 "
        f"-c {N_CTX} -fa -ctk q8_0 -ctv q8_0 "
        f"--host 127.0.0.1 --port {PORT}"
    )
    log(f"启动 llama-server: {cmd}")
    proc = subprocess.Popen(cmd, shell=True)
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(120):                 # 最多等 ~10 分钟
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    log("llama-server HEALTHY")
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server 进程提前退出")
        time.sleep(5)
    raise RuntimeError("llama-server 启动超时")


def chat(prompt: str) -> str:
    """调用 OpenAI 兼容接口。"""
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        **SAMPLING,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def gpu_info() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cuda unavailable"
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        free = [round(torch.cuda.mem_get_info(i)[0] / 1024 ** 3, 1)
                for i in range(torch.cuda.device_count())]
        return f"devices={names} free_GB={free}"
    except Exception as exc:                        # noqa: BLE001
        return f"torch 不可用: {exc}"


def main() -> None:
    started = datetime.now(timezone.utc)
    notify(f"[Kaggle] Qwen3-30B-A3B 推理开始 {started:%Y-%m-%d %H:%M} UTC")
    os.makedirs(OUT_DIR, exist_ok=True)

    log(f"GPU: {gpu_info()}")
    install_stack()
    gguf = locate_gguf()
    log(f"GGUF: {gguf} ({os.path.getsize(gguf)/1024**3:.1f} GB)")

    server = start_server(gguf)
    try:
        # -------- 你的批量任务：读 tasks.jsonl，逐条推理，落盘 --------
        results = []
        if os.path.exists(TASKS_PATH):
            with open(TASKS_PATH, encoding="utf-8") as f:
                tasks = [json.loads(line) for line in f if line.strip()]
        else:
            log(f"未找到 {TASKS_PATH}，跑一条冒烟测试")
            tasks = [{"id": "smoke", "prompt": "用一句话介绍你自己。"}]

        for i, task in enumerate(tasks, 1):
            try:
                answer = chat(task["prompt"])
                results.append({"id": task.get("id", i),
                                "prompt": task["prompt"], "answer": answer})
                log(f"[{i}/{len(tasks)}] ok id={task.get('id', i)}")
            except Exception as exc:                # noqa: BLE001
                log(f"[{i}/{len(tasks)}] fail: {exc}")
                results.append({"id": task.get("id", i), "error": str(exc)})

        out_path = os.path.join(OUT_DIR, "results.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 把临时权重清掉，避免占满 20GB working 盘
        shutil.rmtree("/kaggle/tmp/gguf", ignore_errors=True)
        log(f"结果已写 {out_path}，共 {len(results)} 条")
    finally:
        server.terminate()

    cost = (datetime.now(timezone.utc) - started).total_seconds() / 60
    notify(f"[Kaggle] 推理完成，{len(results)} 条，用时 {cost:.1f} 分钟")


if __name__ == "__main__":
    main()
