"""Icon detector: OmniParser v2's YOLOv8 model (AGPL-3.0, downloaded separately) run with onnxruntime.

Finds clickable icons and controls that carry no text (hamburger menus, toolbar buttons, close X),
so they get numbered boxes like OCR'd text does. Set up with scripts/export_detector.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "icon_detect" / "model.onnx"


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain non-maximum suppression. boxes: (N, 4) x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class IconDetector:
    def __init__(self, path: str | Path = DEFAULT_PATH, conf: float = 0.15, iou: float = 0.3,
                 max_box_frac: float = 0.25, providers: list[str] | None = None):
        import onnxruntime as ort

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/export_detector.py first")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), opts, providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size = self.session.get_inputs()[0].shape[2]
        self.conf, self.iou, self.max_box_frac = conf, iou, max_box_frac

    def _letterbox(self, image: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
        w, h = image.size
        r = self.size / max(w, h)
        nw, nh = round(w * r), round(h * r)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(image.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
        arr = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
        return arr, r, (px, py)

    def detect(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], float]]:
        arr, r, (px, py) = self._letterbox(image)
        out = self.session.run(None, {self.input: arr})[0][0]  # (5, N): cx, cy, w, h, score
        scores = out[4]
        keep = scores >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        scores = scores[keep]
        idx = nms(boxes, scores, self.iou)
        W, H = image.size
        result = []
        for i in idx:
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if (x2 - x1) * (y2 - y1) > self.max_box_frac * W * H or x2 - x1 < 4 or y2 - y1 < 4:
                continue  # whole-panel boxes are not click targets
            result.append(((int(x1), int(y1), int(x2), int(y2)), float(scores[i])))
        return result


def load_detector(path: str | Path = DEFAULT_PATH) -> IconDetector | None:
    """The detector is optional: without the model file the agent falls back to OCR + point/zoom."""
    try:
        return IconDetector(path)
    except (FileNotFoundError, ImportError):
        return None
