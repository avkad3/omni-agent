import pytest

from omni_agent.actions import Action, extract_json, is_risky, parse_action


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"action": "click"}\n```') == {"action": "click"}
    assert extract_json('I will do this: {"action": "type", "text": "a {b}"} ok') == {"action": "type", "text": "a {b}"}
    assert extract_json('[{"action": "done"}]') == {"action": "done"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_action_aliases_and_keys():
    a = parse_action('{"action": "press", "key": "ctrl+s"}')
    assert a.action == "hotkey" and a.keys == ["ctrl", "s"]
    a = parse_action('{"type": "left_click", "element_id": "12", "target": "Save"}')
    assert a.action == "click" and a.element_id == 12
    assert parse_action('{"action": "click", "element_id": null, "target": "x"}').element_id is None
    assert parse_action('{"action": "Finish", "summary": "ok"}').action == "done"


def test_parse_action_rejects_unknown():
    with pytest.raises(ValueError):
        parse_action('{"action": "teleport"}')


@pytest.mark.parametrize(
    "action,label,risky",
    [
        (Action(action="click", target="Delete"), "", True),
        (Action(action="click", target="OK"), "Send", True),
        (Action(action="click", target="OK"), "OK", False),
        (Action(action="hotkey", keys=["shift", "delete"]), "", True),
        (Action(action="hotkey", keys=["ctrl", "s"]), "", False),
        (Action(action="type", text="hello world"), "", False),
        (Action(action="launch", text="Delete Tool"), "", False),
    ],
)
def test_risky(action, label, risky):
    assert is_risky(action, label) is risky
