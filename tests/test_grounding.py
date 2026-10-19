from PIL import Image

from omni_agent.grounding import Grounder, best_text_match, parse_point, text_score
from omni_agent.perception.elements import Element, annotate, describe


def els():
    return [
        Element(0, "text", (10, 10, 60, 30), "File"),
        Element(1, "text", (70, 10, 120, 30), "Edit"),
        Element(2, "text", (200, 300, 600, 330), "Remember to save your work before closing"),
        Element(3, "text", (700, 500, 760, 530), "Save"),
        Element(4, "text", (780, 500, 860, 530), "Don't save"),
        Element(5, "icon", (900, 10, 930, 40)),
    ]


def test_text_score():
    assert text_score("the Save button", "Save") == 1.0
    assert text_score('"Edit"', "Edit") == 1.0
    assert text_score("save", "Don't save") < 1.0


def test_best_text_match_prefers_exact_short_label():
    el, s = best_text_match("Save button", els())
    assert el.id == 3 and s == 1.0
    el, _ = best_text_match("click 'Don't save'", els())
    assert el.id == 4


class FakeModel:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt, images, system="", json_mode=True):
        self.calls += 1
        return self.replies.pop(0)


def test_grounder_text_needs_no_model():
    m = FakeModel([])
    g = Grounder(m, (1288, 728)).ground(Image.new("RGB", (1920, 1080)), els(), "Save")
    assert g.method == "text" and (g.x, g.y) == (730, 515) and m.calls == 0


def test_grounder_text_overrules_wrong_id():
    g = Grounder(None, (1288, 728)).ground(Image.new("RGB", (1920, 1080)), els(), "Edit", element_id=0)
    assert g.method == "text" and g.element.id == 1


def test_grounder_uses_mark_for_icons():
    g = Grounder(None, (1288, 728)).ground(Image.new("RGB", (1920, 1080)), els(), "settings gear", element_id=5)
    assert g.method == "mark" and g.element.id == 5


def test_grounder_point_with_zoom_maps_back_to_frame():
    # First guess in model space (644, 364) = frame centre (960, 540). The zoom crop is 448px around it,
    # shown at 896px, and the model answers its centre (448, 448) -> frame (960, 540).
    m = FakeModel(['{"x": 644, "y": 364}', '{"x": 448, "y": 448}'])
    g = Grounder(m, (1288, 728)).ground(Image.new("RGB", (1920, 1080)), els(), "the red bird")
    assert g.method == "point+zoom" and abs(g.x - 960) < 1 and abs(g.y - 540) < 1 and m.calls == 2


def test_parse_point_formats():
    assert parse_point('{"x": 10, "y": 20}', 100, 100) == (10, 20)
    assert parse_point('{"bbox_2d": [10, 10, 30, 50]}', 100, 100) == (20, 30)
    assert parse_point('{"x": 0.5, "y": 0.25}', 200, 100) == (100, 25)
    assert parse_point('{"x": 500, "y": 500}', 200, 100) == (100, 50)  # 0-1000 normalised
    assert parse_point("the element is at (12, 34)", 100, 100) == (12, 34)
    assert parse_point('{"x": -1, "y": -1}', 100, 100) is None


def test_annotate_and_describe():
    img = annotate(Image.new("RGB", (1920, 1080)), els(), scale=0.5)
    assert img.size == (1920, 1080)
    text = describe(els(), (1920, 1080))
    assert '[3] text "Save" at (0.38,0.48)' in text and "[5] icon <icon>" in text


def test_split_line_separates_menu_words():
    from omni_agent.perception.ocr import split_line

    words = [("File", (55, 16, 80, 30)), ("Edit", (98, 16, 125, 30)), ("Save", (300, 16, 330, 30)),
             ("as", (334, 16, 348, 30))]
    out = split_line(words)
    assert [e.text for e in out] == ["File", "Edit", "Save as"]


def test_snap_prefers_containing_then_nearest():
    from omni_agent.grounding import snap

    boxes = [Element(0, "icon", (100, 100, 140, 140)), Element(1, "icon", (90, 90, 300, 300)),
             Element(2, "icon", (500, 500, 540, 540))]
    assert snap(120, 120, boxes).id == 0  # smallest box containing the point
    assert snap(530, 560, boxes).id == 2  # 20 px below box 2
    assert snap(800, 800, boxes) is None


def test_point_snaps_to_icon_without_zoom():
    # Model guesses (60, 30) in 1288x728 space = (89, 45) on 1920x1080, inside the hamburger icon box.
    m = FakeModel(['{"x": 60, "y": 30}'])
    icons = [Element(0, "icon", (75, 30, 105, 60))]
    g = Grounder(m, (1288, 728)).ground(Image.new("RGB", (1920, 1080)), icons, "menu button")
    assert g.method == "point+snap" and (g.x, g.y) == (90, 45) and m.calls == 1
