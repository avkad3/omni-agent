"""One-time setup: download OmniParser v2's icon detector (YOLOv8, AGPL-3.0) and export it to ONNX.

Run in a throwaway environment that has `ultralytics` (it pulls in PyTorch). The agent itself only needs
onnxruntime at run time.

    python scripts/export_detector.py models/icon_detect
"""

import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/"


def main(out_dir: str = "models/icon_detect") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "model.yaml", "train_args.yaml", "LICENSE"):
        if not (out / name).exists():
            urllib.request.urlretrieve(BASE + name, out / name)
    from ultralytics import YOLO

    onnx = YOLO(str(out / "model.pt")).export(format="onnx", imgsz=1280, dynamic=False, simplify=True, opset=17)
    print("exported", onnx)


if __name__ == "__main__":
    main(*sys.argv[1:])
