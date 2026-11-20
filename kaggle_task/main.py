"""Kaggle 侧实际跑的任务：自检 GPU -> 干正事 -> webhook 通知。"""
import os
import json
import platform
import urllib.request
from datetime import datetime, timezone

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")


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


def main() -> None:
    started = datetime.now(timezone.utc)
    notify(f"[Kaggle] 任务开始 {started:%Y-%m-%d %H:%M} UTC")

    # 1) GPU 自检
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "N/A"
    except Exception as exc:                       # noqa: BLE001
        gpu_ok, gpu_name = False, f"torch 不可用: {exc}"

    notify(f"[Kaggle] GPU={gpu_ok} 设备={gpu_name} py={platform.python_version()}")

    # 2) 这里换成你真正的训练/推理逻辑
    #    ...
    #    产物一律写到 ./output，别写 /kaggle/input（只读）
    #    ...

    finished = datetime.now(timezone.utc)
    cost = (finished - started).total_seconds() / 60
    notify(f"[Kaggle] 任务完成，用时 {cost:.1f} 分钟")


if __name__ == "__main__":
    main()
