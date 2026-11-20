"""Kaggle 2×T4 上用 llama.cpp 跑 Qwen3-30B-A3B-GGUF 定时推理。

数据集解析优先级：
  1) /kaggle/input/你的用户名__qwen30b-gguf-cache/  （私有缓存，省配额）
  2) /kaggle/input/Qwen3-30B-A3B-GGUF/              （公开数据集直挂）
  3) 现下到 /kaggle/tmp                              （回退，费时）
任务文件 tasks.jsonl 同理：优先私有数据集，缺省跑内置冒烟测试。
"""
import os, re, json, glob, shutil, subprocess, urllib.request
from datetime import datetime, timezone

# ---------------- 配置（可用环境变量覆盖） ----------------
GGUF_REPO      = os.environ.get("GGUF_REPO", "unsloth/Qwen3-30B-A3B-GGUF")
GGUF_QUANT     = os.environ.get("GGUF_QUANT", "Q4_K_M")     # OOM 时换 Q3_K_L
N_CTX          = int(os.environ.get("N_CTX", "4096"))
PORT           = int(os.environ.get("PORT", "8080"))
OUT_DIR        = os.environ.get("OUT_DIR", "/kaggle/working/out")
TMP_DIR        = "/kaggle/tmp/m"
WEBHOOK_URL    = os.environ.get("WEBHOOK_URL", "")

# 私有数据集 slug（你在 Output→New Dataset 时填的 Title）；挂载后目录名是 用户__slug
CACHE_SLUG     = os.environ.get("CACHE_SLUG", "你的用户名__qwen30b-gguf-cache")
TASKS_DIR_SLUG = os.environ.get("TASKS_DIR_SLUG", "你的用户名__my-tasks")

# ---------------- 通知 ----------------
def notify(text: str) -> None:
    if not WEBHOOK_URL:
        return
    payload = json.dumps({"msgtype": "text", "text": {"content": text}}).encode()
    req = urllib.request.Request(WEBHOOK_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

# ---------------- 凭证兜底：Secrets 优先，取不到就用环境里已有的 ----------------
def load_secrets() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        sc = UserSecretsClient()
        for name in ("WEBHOOK_URL",):
            if not os.environ.get(name):
                try:
                    v = sc.get_secret(name)
                    if v:
                        os.environ[name] = v
                except Exception:
                    pass
        globals()["WEBHOOK_URL"] = os.environ.get("WEBHOOK_URL", "")
    except Exception:
        pass

# ---------------- 数据集解析 ----------------
def _pick_gguf_in(directory: str) -> str:
    """在给定目录里挑一个匹配的 gguf：量化档优先，其次最大文件。"""
    if not directory or not os.path.isdir(directory):
        return ""
    pats = [f"*{GGUF_QUANT}*.gguf", "*.gguf"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(directory, "**", p), recursive=True))
        if hits:
            return hits[0]
    return ""

def resolve_gguf() -> str:
    """1) 私有缓存  2) 公开数据集直挂  3) 现下。"""
    input_root = "/kaggle/input"
    candidates = []
    if os.path.isdir(input_root):
        # 私有缓存优先：目录名精确匹配 slug 或双下划线变体
        for d in sorted(os.listdir(input_root)):
            full = os.path.join(input_root, d)
            if not os.path.isdir(full):
                continue
            if d == CACHE_SLUG or d.replace("__", "/") == CACHE_SLUG:
                candidates.append(full)
        # 其次任何含 GGUF 的目录（含公开数据集）
        for d in sorted(os.listdir(input_root)):
            full = os.path.join(input_root, d)
            if os.path.isdir(full) and full not in candidates:
                candidates.append(full)
    for d in candidates:
        hit = _pick_gguf_in(d)
        if hit:
            print(f"[gguf] 命中挂载数据集: {hit}")
            return hit

    # 回退：现下
    print(f"[gguf] 未挂载数据集，回退下载 {GGUF_REPO} ({GGUF_QUANT}) …")
    os.makedirs(TMP_DIR, exist_ok=True)
    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=GGUF_REPO, allow_patterns=[f"*{GGUF_QUANT}*.gguf"],
                      local_dir=TMP_DIR)
    hit = _pick_gguf_in(TMP_DIR)
    if not hit:
        raise FileNotFoundError("下载后仍找不到 gguf")
    return hit

def resolve_tasks() -> str:
    """任务文件：私有数据集优先，缺省返回空串（走冒烟测试）。"""
    input_root = "/kaggle/input"
    if os.path.isdir(input_root):
        for d in sorted(os.listdir(input_root)):
            full = os.path.join(input_root, d)
            if not os.path.isdir(full):
                continue
            if d == TASKS_DIR_SLUG or d.replace("__", "/") == TASKS_DIR_SLUG:
                hits = sorted(glob.glob(os.path.join(full, "**", "*.jsonl"),
                                        recursive=True))
                if hits:
                    print(f"[tasks] 命中: {hits[0]}")
                    return hits[0]
    # 兜底：任何目录里的 tasks*.jsonl
    if os.path.isdir(input_root):
        for d in sorted(os.listdir(input_root)):
            hits = sorted(glob.glob(os.path.join(input_root, d, "**", "tasks*.jsonl"),
                                    recursive=True))
            if hits:
                print(f"[tasks] 兜底命中: {hits[0]}")
                return hits[0]
    return ""

# ---------------- 推理服务 ----------------
def ensure_llama_server() -> None:
    try:
        subprocess.run("llama-server --version", shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except Exception:
        pass
    print("[setup] 安装 llama-cpp-python (CUDA wheel) …")
    subprocess.run(
        "pip install -q llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128",
        shell=True, check=True)

def start_server(gguf_path: str):
    """2×T4 按层切分；单卡时去掉 tensor-split。"""
    n_gpu_layers = 99
    cmd = (f'llama-server -m "{gguf_path}" -ngl {n_gpu_layers} '
           f'--split-mode layer --tensor-split 1,1 '
           f'-c {N_CTX} -fa -ctk q8_0 -ctv q8_0 '
           f'--host 127.0.0.1 --port {PORT}')
    print("[server]", cmd)
    proc = subprocess.Popen(cmd, shell=True)
    url = f"http://127.0.0.1:{PORT}/health"
    for _ in range(120):                       # 最多等约 10 分钟加载
        try:
            if urllib.request.urlopen(url, timeout=5).status == 200:
                print("[server] HEALTHY")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server 退出，码={proc.returncode}")
        import time; time.sleep(5)
    raise TimeoutError("llama-server 启动超时")

def chat(prompt: str) -> str:
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

# ---------------- 主流程 ----------------
def main() -> None:
    load_secrets()
    started = datetime.now(timezone.utc)
    notify(f"[Kaggle] Qwen3-30B-A3B 推理开始 {started:%Y-%m-%d %H:%M} UTC")
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import torch
        assert torch.cuda.is_available(), "没分到 GPU，检查 Accelerator 设置"
        n_dev = torch.cuda.device_count()
        dev = [torch.cuda.get_device_name(i) for i in range(n_dev)]
        notify(f"[Kaggle] GPU x{n_dev}: {dev}")
        if n_dev < 2:
            print("[warn] 只有 1 张卡，去掉 tensor-split 更稳")
    except Exception as e:
        raise RuntimeError(f"GPU 自检失败: {e}")

    gguf_path = resolve_gguf()
    print(f"[gguf] 使用 {gguf_path} "
          f"({os.path.getsize(gguf_path)/1e9:.1f} GB)")
    ensure_llama_server()
    proc = start_server(gguf_path)

    try:
        tasks_path = resolve_tasks()
        if tasks_path:
            tasks = [json.loads(l) for l in open(tasks_path, encoding="utf-8") if l.strip()]
        else:
            print("[tasks] 未挂载任务数据，跑内置冒烟测试")
            tasks = [{"id": "smoke", "prompt": "用一句话介绍你自己。"}]

        results = []
        for t in tasks:
            tid = t.get("id", "task")
            prompt = t.get("prompt", "")
            if not prompt:
                continue
            try:
                ans = chat(prompt)
            except Exception as e:
                ans = f"[ERROR] {e}"
            results.append({"id": tid, "prompt": prompt, "output": ans})
            print(f"[{tid}] {ans[:120]}")

        out_file = os.path.join(OUT_DIR, "results.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        notify(f"[Kaggle] 完成 {len(results)} 条，结果 {out_file}，用时 "
               f"{(datetime.now(timezone.utc)-started).total_seconds()/60:.1f} 分钟")
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
