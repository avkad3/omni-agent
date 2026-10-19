import numpy as np
from PIL import Image

from omni_agent.replay import TrajectoryStore, TrajStep, crop_patch, find_patch, goal_key


def noisy(seed=0, size=(1920, 1080)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_find_patch_relocates_a_moved_target():
    img = noisy()
    patch = crop_patch(img, 700, 400)  # 96x96 around (700, 400)
    moved = Image.new("RGB", img.size, "white")
    moved.paste(img.crop((0, 0, 1800, 1000)), (60, 40))  # whole screen shifted by (+60, +40)
    x, y, score = find_patch(moved, patch, near=(700, 400))
    assert abs(x - 760) <= 2 and abs(y - 440) <= 2 and score > 0.9


def test_find_patch_rejects_flat_patch_and_absent_target():
    blank = Image.new("RGB", (800, 600), "white")
    assert find_patch(blank, crop_patch(blank, 400, 300), near=(400, 300)) is None
    _, _, score = find_patch(noisy(1), crop_patch(noisy(2), 300, 300), near=(300, 300))
    assert score < 0.5


def test_store_round_trip(tmp_path):
    store = TrajectoryStore(tmp_path)
    steps = [(TrajStep(0, {"action": "launch", "text": "Calculator"}), None),
             (TrajStep(1, {"action": "click", "target": "x"}, rel_xy=(0.5, 0.5)), noisy().crop((0, 0, 96, 96)))]
    store.save("Open  Calculator", ["open Calculator", "click x"], steps)
    t = store.load("open calculator")  # case and spacing do not matter
    assert t.plan == ["open Calculator", "click x"] and t.steps[1].patch == "patch_01.png"
    assert store.patch("open calculator", t.steps[1]).size == (96, 96)
    assert goal_key("Open Calculator") == goal_key("open   calculator")
