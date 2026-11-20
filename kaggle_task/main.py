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
import glob
import sys
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
# 可选：把 GGUF 存成 Kaggle Dataset 挂载，直接复用，省掉每次 ~80s 下载
GGUF_CACHE_DIR = os.environ.get("GGUF_CACHE_DIR", "/kaggle/input/qwen30b-gguf-cache")
# 可选：把编好的 CUDA wheel 存成 Kaggle Dataset 挂载，命中即秒装，省掉每次 ~30min 编译
WHEEL_CACHE_DIR = os.environ.get("WHEEL_CACHE_DIR", "/kaggle/input/llama-cpp-cuda-wheel")
# 源码编译时 wheel 的输出目录（/kaggle/working 会随版本保存，可据此创建 Dataset）
WHEEL_BUILD_DIR = "/kaggle/working/wheels"

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


def _gpu_offload_ok() -> bool:
    """自检已装的 llama_cpp 是否带 CUDA offload 支持。"""
    ret = subprocess.run(
        "python -c \"import llama_cpp as L; "
        "f=getattr(L,'llama_supports_gpu_offload',None) or "
        "getattr(getattr(L,'llama_cpp',None),'llama_supports_gpu_offload',None); "
        "print('GPU_OFFLOAD=' + ('1' if (f and f()) else '0'))\"",
        shell=True, capture_output=True, text=True,
    )
    ok = "GPU_OFFLOAD=1" in (ret.stdout or "")
    log(f"GPU offload supported: {ok}")
    return ok


def _find_cached_wheel() -> str:
    """在挂载 Dataset 里找预编译 CUDA wheel，找不到返回空串。

    llama-cpp-python 源码编译产物是 py3-none-linux_x86_64（不绑 Python 次版本），
    但预编译 cu128 wheel 是 cp3xx；两种都接受，兼容性交给安装后的自检兜底。
    """
    if not os.path.isdir(WHEEL_CACHE_DIR):
        return ""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"   # 如 cp313
    found = ""
    for root, _dirs, files in os.walk(WHEEL_CACHE_DIR):
        for f in files:
            if not (f.startswith("llama_cpp_python-") and f.endswith(".whl")):
                continue
            if "py3-none" in f or tag in f:        # 源码编译 wheel / 官方预编译 wheel
                found = os.path.join(root, f)
                break
        if found:
            break
    return found


def _install_wheel(path: str) -> None:
    """先卸载旧版再装，确保 CUDA wheel 覆盖 CPU 版；依赖仅按需补装，不升级已有包。

    不用 --force-reinstall：它会连带把 numpy 2.1.3 强制升到 2.5.5，可能破坏
    Kaggle 预装的 torch 等环境。先 uninstall 主包即可保证重装，依赖保持现状。
    """
    subprocess.run("pip uninstall -y llama-cpp-python || true", shell=True, check=True)
    subprocess.run(f"pip install {path}", shell=True, check=True)


def install_stack() -> None:
    """装 CUDA 版 llama-cpp-python：缓存 wheel 优先 → 源码编译兜底。

    Kaggle 默认 Python 3.13 无匹配 cu128 wheel，pip 会回退纯 CPU 源码包，
    因此编译必须 FORCE_CMAKE + CMAKE_ARGS 显式开 CUDA。
    首次编译的 wheel 落到 /kaggle/working/wheels，保存成 Dataset 挂载后，
    后续运行可从 WHEEL_CACHE_DIR 秒装，省掉每次 ~30 分钟编译。
    """
    # 0) 已装好 CUDA 版 → 跳过（只走子进程检测，不在本进程 import，避免污染 sys.modules）
    #    注：本进程一旦 import 过 CPU 版，后面即使重装 CUDA wheel 也仍会拿到旧的缓存模块。
    if _gpu_offload_ok():
        log("llama-cpp-python CUDA 版已就绪，跳过安装")
        return

    # 1) 优先复用挂载 Dataset 里的预编译 wheel
    cached = _find_cached_wheel()
    if cached:
        log(f"[wheel] 命中挂载缓存: {cached}")
        try:
            _install_wheel(cached)
            if _gpu_offload_ok():
                log("[wheel] 秒装成功，跳过源码编译（省约 30 分钟）")
                return
            log("[wheel] 缓存 wheel 无 CUDA 支持，回退源码编译")
        except subprocess.CalledProcessError as exc:
            log(f"[wheel] 缓存 wheel 安装失败: {exc}，回退源码编译")

    # 2) 兜底：源码编译 CUDA wheel（官方写法 + Kaggle stub 修正）
    log("编译安装 llama-cpp-python (CUDA / sm_75) ...")
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
    os.makedirs(WHEEL_BUILD_DIR, exist_ok=True)
    # 用 pip wheel 把编译产物落成 .whl（既安装又留存，供保存为 Dataset 复用）
    cmd = f"pip wheel --no-deps --no-cache-dir --verbose -w {WHEEL_BUILD_DIR} llama-cpp-python"
    with open(build_log, "w", encoding="utf-8", errors="replace") as logf:
        ret = subprocess.run(cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        log("CUDA 编译失败，输出日志尾部（最后 80 行）:")
        subprocess.run(f"tail -80 {build_log}", shell=True)
        raise RuntimeError("llama-cpp-python CUDA 编译失败，见上方日志")
    log("CUDA 编译成功")

    wheels = glob.glob(os.path.join(WHEEL_BUILD_DIR, "llama_cpp_python-*.whl"))
    if not wheels:
        raise RuntimeError(f"编译完成但 {WHEEL_BUILD_DIR} 下未找到 .whl")
    log(f"[wheel] 编译产物: {wheels[0]} ({os.path.getsize(wheels[0])/1024**2:.0f} MB)")
    log(f"[wheel] 把 {WHEEL_BUILD_DIR} 保存成 Kaggle Dataset 并挂载，下次即可秒装")
    _install_wheel(wheels[0])

    # 装完立刻自检：确认 CUDA 后端真的编进去了，别等模型加载才发现跑在 CPU 上
    if not _gpu_offload_ok():
        raise RuntimeError("CUDA 自检未通过，推理会退化到 CPU，请检查编译日志")


def locate_gguf() -> str:
    """定位 GGUF：优先用挂载数据集缓存，否则从 HF 下载。返回本地 .gguf 路径。"""
    # 1) 挂载数据集缓存（/kaggle/input 只读，命中即直接用，无需下载）
    if os.path.isdir(GGUF_CACHE_DIR):
        for root, _dirs, files in os.walk(GGUF_CACHE_DIR):
            for f in files:
                if f.endswith(".gguf") and GGUF_QUANT in f:
                    path = os.path.join(root, f)
                    log(f"[gguf] 命中挂载缓存: {path} ({os.path.getsize(path)/1024**3:.1f} GB)")
                    return path

    # 2) 未命中则从 HuggingFace 下载
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
