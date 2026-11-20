"""Kaggle 2×T4 定时推理：Qwen3-30B-A3B-GGUF（llama.cpp 按层切分）。

用途：GitHub Actions 定时 push 到 Kaggle，Kaggle 侧用 2 张 T4 跑批量推理。
说明：
  - 只做推理，不做训练。GGUF 是 llama.cpp 部署格式，不能用 Unsloth/PEFT 微调；
    要微调请换成 unsloth/Qwen3-30B-A3B（safetensors）另写脚本。
  - 两张 T4 是各自独立的 16GB，用 Python API 的 split_mode=LAYER + tensor_split=[1,1] 按层均摊。
    不用 llama-server CLI（pip 包不提供该二进制），直接用 llama_cpp.Llama + create_chat_completion。
  - Q4_K_M 文件约 17.3GB + KV，2×T4 合计 32GB 能塞下；
    若 OOM 请把 GGUF_QUANT 换成 Q3_K_L 并把 N_CTX 调到 4096。
  - 产物一律写 /kaggle/working（会随版本保存），别写 /kaggle/input（只读）。
"""
import os
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone

# ---------------- 配置（可用 Kaggle Notebook 环境变量覆盖） ----------------
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/Qwen3-30B-A3B-GGUF")
GGUF_QUANT = os.environ.get("GGUF_QUANT", "Q4_K_M")   # OOM 时改 Q3_K_L
N_CTX = int(os.environ.get("N_CTX", "8192"))          # OOM 时降到 4096
NGL = int(os.environ.get("NGL", "99"))                # 全部层 offload 到 GPU
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
    """装 llama.cpp Python 绑定。强制源码编译并启用 CUDA（官方写法）。

    Kaggle 默认 Python 3.13 没有匹配的 cu128 预编译 wheel，
    pip 会回退到纯 CPU 源码包；必须 FORCE_CMAKE + CMAKE_ARGS 显式开 CUDA。
    """
    log("编译安装 llama-cpp-python (CUDA) ...")
    # 环境诊断：CUDA 编译失败多半是 nvcc / cmake 缺失，先打印出来
    subprocess.run(
        "echo '--- nvcc ---'; which nvcc && nvcc --version | tail -4; "
        "echo '--- cmake ---'; cmake --version | head -1; "
        "echo '--- CUDA_HOME ---'; echo \"${CUDA_HOME:-<unset>}\"; "
        "echo '--- python ---'; python --version",
        shell=True,
    )
    env = os.environ.copy()
    env["FORCE_CMAKE"] = "1"
    env.setdefault("CUDA_HOME", "/usr/local/cuda")

    # 显式告诉 CMake driver stub 位置，让 CUDA::cuda_driver target 能被找到
    stub = "/usr/local/cuda/lib64/stubs/libcuda.so"
    cmake_args = "-DGGML_CUDA=on -DGGML_CUDA_NO_VMM=ON"
    if os.path.exists(stub):
        cmake_args += f" -DCUDA_CUDA_LIBRARY={stub}"
    env["CMAKE_ARGS"] = cmake_args

    # verbose 日志落盘，失败时回显尾部——否则 pip 只给 "No available output"
    build_log = "/kaggle/tmp/pip_llama_build.log"
    os.makedirs(os.path.dirname(build_log), exist_ok=True)
    cmd = "pip install --no-cache-dir --verbose llama-cpp-python"
    with open(build_log, "w", encoding="utf-8", errors="replace") as logf:
        ret = subprocess.run(cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        log("CUDA 编译失败，输出日志尾部（最后 80 行）:")
        subprocess.run(f"tail -80 {build_log}", shell=True)
        raise RuntimeError("llama-cpp-python CUDA 编译失败，见上方日志")
    log("CUDA 编译成功")

    # 装完立刻自检：确认 CUDA 后端真的编进去了，别等模型加载才发现跑在 CPU 上
    subprocess.run(
        "python -c \"import llama_cpp as L; "
        "f=getattr(L,'llama_supports_gpu_offload',None) or "
        "getattr(getattr(L,'llama_cpp',None),'llama_supports_gpu_offload',None); "
        "print('GPU offload supported:', f() if f else 'unknown')\"",
        shell=True,
    )


def locate_gguf() -> str:
    """下载 GGUF 并返回本地 .gguf 文件路径。"""
    log(f"下载 {GGUF_REPO}:{GGUF_QUANT} ...")
    # huggingface-cli 已废弃且不再工作，改用 hf；新版已移除 --local-dir-use-symlinks
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


def load_model(gguf_path: str):
    """用 llama-cpp-python 的 Python API 直接加载模型（无需 llama-server 二进制）。

    两卡切分：split_mode=LAYER + tensor_split=[1,1]；Flash Attention 开启。
    KV cache 用默认 f16（2×T4 余量足够），避免猜枚举值导致加载失败。
    """
    import llama_cpp

    log(f"加载模型: ngl={NGL} split=layer tensor=1,1 ctx={N_CTX} flash_attn=True")
    return llama_cpp.Llama(
        model_path=gguf_path,
        n_gpu_layers=NGL,
        split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        tensor_split=[1.0, 1.0],
        n_ctx=N_CTX,
        flash_attn=True,
        verbose=True,
    )


def chat(llm, prompt: str) -> str:
    """直接调用 Python API 的 OpenAI 兼容方法。"""
    data = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        **SAMPLING,
    )
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

    llm = load_model(gguf)
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
                answer = chat(llm, task["prompt"])
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
        del llm  # 释放 GGUF 与显存

    cost = (datetime.now(timezone.utc) - started).total_seconds() / 60
    notify(f"[Kaggle] 推理完成，{len(results)} 条，用时 {cost:.1f} 分钟")


if __name__ == "__main__":
    main()
