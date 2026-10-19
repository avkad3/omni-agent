"""UI elements found on screen, and the numbered-box ("set-of-mark") view given to the model."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Element:
    id: int
    kind: str  # "text" or "icon"
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    text: str = ""
    score: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


_PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (240, 50, 230)]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image: Image.Image, elements: list[Element], scale: float = 1.0) -> Image.Image:
    """Draw each element's box and id. `scale` maps frame pixels to the image being drawn on."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(max(11, int(14 * max(scale, 0.7))))
    for el in elements:
        color = _PALETTE[el.id % len(_PALETTE)]
        x1, y1, x2, y2 = (int(v * scale) for v in el.box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = str(el.id)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ly = y1 - th - 2 if y1 - th - 2 >= 0 else y2 + 1
        draw.rectangle((x1, ly, x1 + tw + 4, ly + th + 2), fill=color)
        draw.text((x1 + 2, ly), label, fill="white", font=font)
    return out


def describe(elements: list[Element], frame_size: tuple[int, int], limit: int = 120) -> str:
    """Compact text list for the prompt. Positions are given as screen fractions so the list
    stays meaningful whatever resolution the model sees."""
    w, h = frame_size
    lines = []
    for el in elements[:limit]:
        cx, cy = el.center
        where = f"({cx / w:.2f},{cy / h:.2f})"
        label = f'"{el.text}"' if el.text else "<icon>"
        lines.append(f"[{el.id}] {el.kind} {label} at {where}")
    if len(elements) > limit:
        lines.append(f"... {len(elements) - limit} more not listed")
    return "\n".join(lines)
