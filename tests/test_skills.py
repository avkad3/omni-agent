import pytest

from omni_agent.skills import direct_action


@pytest.mark.parametrize(
    "step,expected",
    [
        ("open Notepad", ("launch", {"text": "Notepad"})),
        ("Launch the Calculator app", ("launch", {"text": "Calculator"})),
        ('type "Hello, world"', ("type", {"text": "Hello, world"})),
        ("type 12*7", ("type", {"text": "12*7"})),
        ("type into the search box 'cats'", ("type", {"text": "cats"})),
        ("press ctrl+s", ("hotkey", {"keys": ["ctrl", "s"]})),
        ("press Enter", ("hotkey", {"keys": ["enter"]})),
        ("press alt + f4", ("hotkey", {"keys": ["alt", "f4"]})),
        ("wait 2 seconds", ("wait", {"seconds": 2.0})),
        ("scroll down", ("scroll", {"direction": "down"})),
        ("click Save", ("click", {"target": "Save"})),
        ("click on the 'File' menu", ("click", {"target": "File"})),
        ("double-click Recycle Bin", ("double_click", {"target": "Recycle Bin"})),
        ("right click the desktop", ("right_click", {"target": "the desktop"})),
    ],
)
def test_direct_action(step, expected):
    a = direct_action(step)
    assert a is not None and a.action == expected[0]
    for k, v in expected[1].items():
        assert getattr(a, k) == v


@pytest.mark.parametrize("step", ["make the text bigger", "find the cheapest flight", "press", "type"])
def test_needs_model(step):
    assert direct_action(step) is None


def test_make_safe_plan_inserts_new_document():
    from omni_agent.skills import make_safe_plan

    fresh = ["open Notepad", "press ctrl+n", "wait 1 seconds", 'type "x"']
    assert make_safe_plan(["open Notepad", 'type "x"']) == fresh
    assert make_safe_plan(["open Calculator", 'type "1+1"']) == ["open Calculator", 'type "1+1"']
    assert make_safe_plan(["open Notepad", "click File"]) == ["open Notepad", "click File"]


@pytest.mark.parametrize(
    "title,goal,expected",
    [
        ("recovery-codes.txt - Notepad", "type hello", "recovery-codes.txt"),
        ("*notes.md - Notepad", "type hello", "notes.md"),
        ("notes.md - Notepad", "add a line to notes.md", None),
        ("Untitled - Notepad", "type hello", None),
        ("Calculator", "type 1+1", None),
        ("Document1 - Word", "type hello", None),
    ],
)
def test_existing_file_in_title(title, goal, expected):
    from omni_agent.skills import existing_file_in_title

    assert existing_file_in_title(title, goal) == expected
