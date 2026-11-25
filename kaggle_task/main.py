"""Kaggle 2×T4 定时推理：Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF（llama-server 二进制 + HTTP）。

用途：GitHub Actions 定时 push 到 Kaggle，Kaggle 侧用 2 张 T4 跑批量推理。
说明：
  - 只做推理，不做训练。GGUF 是 llama.cpp 部署格式，不能用 Unsloth/PEFT 微调。
  - 用挂载 Dataset 里的预编译 llama-server 二进制（含 CUDA .so），起本地 HTTP 服务，
    通过 OpenAI 兼容 /v1/chat/completions 推理；不再编译/安装 llama-cpp-python wheel。
  - 两张 T4 各自独立 16GB，用 --split-mode layer --tensor-split 1,1 按层均摊。
  - MTP 投机解码：若二进制支持 --mtp/--draft-mtp 则自动开启，启动失败自动回退到无 MTP。
  - Q4_K 文件约 21.7GB + KV，2×T4 合计 32GB 可容纳但余量变小；
    OOM 时把 GGUF_QUANT 换成 Q3_K（约 17.2GB）并把 N_CTX 调到 4096。
  - 产物一律写 /kaggle/working（会随版本保存），别写 /kaggle/input（只读）。
"""
import os
import json
import time
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

# ---------------- 配置（可用 Kaggle Notebook 环境变量覆盖） ----------------
GGUF_REPO = os.environ.get("GGUF_REPO", "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF")
GGUF_QUANT = os.environ.get("GGUF_QUANT", "Q4_K")   # OOM 时改 Q3_K
N_CTX = int(os.environ.get("N_CTX", "8192"))          # OOM 时降到 4096
NGL = int(os.environ.get("NGL", "99"))                # 全部层 offload 到 GPU
RUN_LIMIT_HOURS = float(os.environ.get("RUN_LIMIT_HOURS", "3"))
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")  # 如 yongyong-kaggle-xxxx；留空则不通知
# 对应 kernel-metadata.json 的 dataset_sources 占位符 TODO_USER/TODO_TASKS_DATASET；
# 替换 slug 后这里也要同步（或用环境变量覆盖）
TASKS_PATH = os.environ.get("TASKS_PATH", "/kaggle/input/TODO_TASKS_DATASET/tasks.jsonl")
OUT_DIR = "/kaggle/working/out"
# 与 dataset_sources 中 qwen36-35b-a3b-abliterated-mtp-gguf 的挂载目录一致
GGUF_CACHE_DIR = os.environ.get("GGUF_CACHE_DIR", "/kaggle/input/qwen36-35b-a3b-abliterated-mtp-gguf")

# llama-server 二进制版
BIN_DIR = os.environ.get("LLAMA_BIN_DIR", "/kaggle/input/llama-cpp-bin")
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
SERVER_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", "600"))  # 等待 /health 的秒数
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "1800"))           # 单条推理超时（秒）
ENABLE_MTP = os.environ.get("ENABLE_MTP", "1") != "0"  # 1 自动开启（若二进制支持），0 强制关闭
EXTRA_SERVER_ARGS = os.environ.get("EXTRA_SERVER_ARGS", "")  # 追加到 llama-server 的参数，空格分隔

API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
SERVER_LOG = "/kaggle/tmp/llama-server.log"
BIN_WORK = "/kaggle/tmp/llama-bin"  # 二进制复制到可写目录再 chmod +x

# 思考模式官方推荐采样参数；不要用贪心解码
SAMPLING = {"temperature": 0.6, "top_p": 0.95}


def notify(text: str) -> None:
    """ntfy.sh 通知，best-effort；失败不影响主流程。"""
    if not NTFY_TOPIC:
        return
    from urllib.parse import quote

    url = f"{NTFY_SERVER}/{quote(NTFY_TOPIC, safe='')}"
    req = urllib.request.Request(
        url,
        data=text.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8",
                 "Title": "Kaggle GPU job", "Tags": "robot"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


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


def locate_gguf() -> str:
    """定位 GGUF：优先挂载 Dataset 缓存，否则从 HF 下载。返回本地 .gguf 路径。"""
    if os.path.isdir(GGUF_CACHE_DIR):
        for root, _dirs, files in os.walk(GGUF_CACHE_DIR):
            for f in files:
                if f.endswith(".gguf") and GGUF_QUANT in f:
                    path = os.path.join(root, f)
                    log(f"[gguf] 命中挂载缓存: {path} ({os.path.getsize(path)/1024**3:.1f} GB)")
                    return path

    log(f"下载 {GGUF_REPO}:{GGUF_QUANT} ...")
    # huggingface-cli 已废弃，改用 hf；新版已移除 --local-dir-use-symlinks
    subprocess.run(
        f"hf download {GGUF_REPO} "
        f"--include '*{GGUF_QUANT}*' --local-dir /kaggle/tmp/gguf",
        shell=True, check=True,
    )
    for root, _dirs, files in os.walk("/kaggle/tmp/gguf"):
        for f in files:
            if f.endswith(".gguf"):
                return os.path.join(root, f)
    raise RuntimeError("未找到 .gguf 文件，检查 GGUF_QUANT 拼写")


def prepare_server_bin() -> str:
    """在挂载 Dataset 里找 llama-server，复制到可写目录并 chmod +x。

    /kaggle/input 只读且文件可能无执行位；.so 与二进制同目录，整体复制后
    通过 LD_LIBRARY_PATH 指向该目录，避免运行时找不到共享库。
    """
    if not os.path.isdir(BIN_DIR):
        raise RuntimeError(f"未找到二进制目录 {BIN_DIR}，检查 LLAMA_BIN_DIR / dataset_sources")

    src_bin = ""
    for root, _dirs, files in os.walk(BIN_DIR):
        for f in files:
            # 排除 libllama-server-impl.so 等库文件，只认可执行主程序
            if f == "llama-server" or (f.startswith("llama-server") and not f.endswith((".so", ".json", ".txt"))):
                src_bin = os.path.join(root, f)
                break
        if src_bin:
            break
    if not src_bin:
        raise RuntimeError(f"{BIN_DIR} 下未找到 llama-server 二进制")

    shutil.rmtree(BIN_WORK, ignore_errors=True)
    os.makedirs(os.path.dirname(BIN_WORK), exist_ok=True)
    shutil.copytree(BIN_DIR, BIN_WORK, dirs_exist_ok=True)
    dst_bin = os.path.join(BIN_WORK, os.path.basename(src_bin))
    # src_bin 可能在 BIN_DIR 子目录里，重新定位复制到 BIN_WORK 后的同名文件
    if not os.path.exists(dst_bin):
        rel = os.path.relpath(src_bin, BIN_DIR)
        dst_bin = os.path.join(BIN_WORK, rel)

    if os.name == "posix":
        os.chmod(dst_bin, 0o755)
        # 同目录 .so 也补执行/读权限（复制后多数已可读）
        for root, _dirs, files in os.walk(BIN_WORK):
            for f in files:
                if f.endswith(".so") or ".so." in f:
                    try:
                        os.chmod(os.path.join(root, f), 0o755)
                    except OSError:
                        pass

    log(f"[bin] 使用 llama-server: {dst_bin} ({os.path.getsize(dst_bin)/1024**2:.0f} MB)")
    return dst_bin


def server_supports(binary: str, flag: str) -> bool:
    """用 --help 探测二进制是否支持某个参数（如 --draft-mtp）。"""
    try:
        ret = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=30,
            env=_server_env(binary),
        )
        return flag in (ret.stdout or "") + (ret.stderr or "")
    except Exception:                                # noqa: BLE001
        return False


def _server_env(binary: str) -> dict:
    env = os.environ.copy()
    bin_dir = os.path.dirname(binary)
    # 优先不带原有 LD_LIBRARY_PATH 的干扰：二进制自带 .so 必须在最前
    env["LD_LIBRARY_PATH"] = bin_dir + (os.pathsep + env["LD_LIBRARY_PATH"]
                                        if env.get("LD_LIBRARY_PATH") else "")
    return env


def start_server(binary: str, gguf: str, use_mtp: bool, minimal: bool = False) -> subprocess.Popen:
    """后台启动 llama-server，日志落盘，返回进程句柄。

    minimal=True 时只保留核心参数，兼容旧版二进制（可能不认 --no-webui / --flash-attn）。
    """
    cmd = [
        binary,
        "-m", gguf,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-c", str(N_CTX),
        "-ngl", str(NGL),
        "--split-mode", "layer",
        "--tensor-split", "1,1",
    ]
    if not minimal:
        cmd += ["--flash-attn", "on", "--no-webui"]
    if use_mtp:
        cmd += ["--mtp", "--draft-mtp"]  # 需模型带 MTP head；失败时调用方回退
    if EXTRA_SERVER_ARGS.strip() and not minimal:
        cmd += EXTRA_SERVER_ARGS.split()

    log(f"[server] 启动: {' '.join(cmd)} (MTP={'on' if use_mtp else 'off'}, minimal={minimal})")
    os.makedirs(os.path.dirname(SERVER_LOG), exist_ok=True)
    logf = open(SERVER_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=_server_env(binary))
    # 让 logf 在父进程保持打开即可，进程结束后由 stop_server 关闭
    return proc


def dump_server_log(n: int = 60) -> None:
    try:
        with open(SERVER_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-n:]
        log(f"[server] 日志尾部（最后 {n} 行）:")
        for line in lines:
            print(line.rstrip(), flush=True)
    except OSError as exc:                           # noqa: BLE001
        log(f"[server] 无法读取日志: {exc}")


def wait_ready(proc: subprocess.Popen, timeout: int) -> bool:
    """轮询 /health；进程退出或超时返回 False。"""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:                            # noqa: BLE001
            pass
        time.sleep(2)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        log("[server] 停止服务 ...")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log("[server] 已退出")


def start_with_fallback(binary: str, gguf: str) -> subprocess.Popen:
    """三级启动：带 MTP → 无 MTP → 最小参数集。每级失败即回退并打日志。"""
    mtp_ok = ENABLE_MTP and server_supports(binary, "--draft-mtp") and server_supports(binary, "--mtp")
    log(f"[mtp] 二进制支持 MTP: {mtp_ok} (ENABLE_MTP={ENABLE_MTP})")

    attempts = [(True, False), (False, False), (False, True)] if mtp_ok else [(False, False), (False, True)]
    for use_mtp, minimal in attempts:
        proc = start_server(binary, gguf, use_mtp, minimal=minimal)
        if wait_ready(proc, SERVER_START_TIMEOUT):
            log(f"[server] 就绪 {HEALTH_URL} (MTP={'on' if use_mtp else 'off'}, minimal={minimal})")
            return proc
        log(f"[server] 启动失败或超时 (MTP={'on' if use_mtp else 'off'}, minimal={minimal})，code={proc.poll()}")
        dump_server_log()
        stop_server(proc)
    raise RuntimeError("llama-server 所有启动方式均失败，见上方日志尾部")


def chat(prompt: str) -> str:
    """通过 llama-server 的 OpenAI 兼容 HTTP API 推理。"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 2048,
        **SAMPLING,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return data["choices"][0]["message"]["content"]


def main() -> None:
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(hours=RUN_LIMIT_HOURS)
    notify(f"[Kaggle] Huihui-Qwen3.6-35B-A3B 推理开始 {started:%Y-%m-%d %H:%M} UTC，"
           f"限时 {RUN_LIMIT_HOURS}h（{deadline:%H:%M} UTC 截止）")
    os.makedirs(OUT_DIR, exist_ok=True)

    log(f"GPU: {gpu_info()}")
    gguf = locate_gguf()
    log(f"GGUF: {gguf} ({os.path.getsize(gguf)/1024**3:.1f} GB)")
    binary = prepare_server_bin()
    proc = start_with_fallback(binary, gguf)

    timed_out = False
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
            # 到点即停：不再发新请求，已完成的先落盘，避免硬杀丢结果
            if datetime.now(timezone.utc) >= deadline:
                timed_out = True
                log(f"达到 {RUN_LIMIT_HOURS}h 时限（已完成 {i-1}/{len(tasks)} 条），停止推理并落盘")
                break
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
        log(f"结果已写 {out_path}，共 {len(results)} 条")
    finally:
        stop_server(proc)                            # 先停服务释放显存
        shutil.rmtree("/kaggle/tmp/gguf", ignore_errors=True)  # 清临时权重，避免占满 working 盘
        shutil.rmtree(BIN_WORK, ignore_errors=True)              # 清复制的二进制

    cost = (datetime.now(timezone.utc) - started).total_seconds() / 60
    tag = "达到时限停止" if timed_out else "正常完成"
    notify(f"[Kaggle] 推理{tag}，{len(results)} 条，用时 {cost:.1f} 分钟")


if __name__ == "__main__":
    main()
