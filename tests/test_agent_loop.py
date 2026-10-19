"""Runs the whole loop with a scripted model, a synthetic screen and a recording executor."""

from PIL import Image

from omni_agent import screen
from omni_agent.agent import Agent
from omni_agent.config import Settings
from omni_agent.perception.elements import Element


class ScriptedModel:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, prompt, images, system="", json_mode=True):
        return self.replies.pop(0)


class FakeParser:
    last_timing = {}

    def parse(self, image):
        return [Element(0, "text", (100, 100, 200, 130), "Notepad"), Element(1, "text", (500, 600, 560, 630), "Save")]


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def point(self, kind, x, y):
        self.calls.append((kind, x, y))

    def type_text(self, text):
        self.calls.append(("type", text))

    def hotkey(self, keys):
        self.calls.append(("hotkey", tuple(keys)))

    def scroll(self, *a):
        self.calls.append(("scroll",))

    def launch(self, app):
        self.calls.append(("launch", app))


def make_agent(tmp_path, replies, ask=lambda q: True):
    s = Settings(runs_dir=tmp_path, action_pause=0, max_steps=6)
    frames = iter(lambda: screen.Frame(Image.new("RGB", (1920, 1080), "white"), left=1920, top=0), None)
    ex = RecordingExecutor()
    agent = Agent(s, model=ScriptedModel(replies), parser=FakeParser(), executor=ex, capture=lambda: next(frames),
                  ask=ask)
    return agent, ex


def test_full_loop(tmp_path):
    agent, ex = make_agent(tmp_path, [
        '{"thought": "open it", "action": "launch", "text": "notepad"}',
        '{"thought": "type", "action": "type", "text": "hello"}',
        '{"thought": "save", "action": "click", "element_id": 0, "target": "Save"}',  # wrong id, text wins
        '{"action": "done", "summary": "saved"}',
    ])
    rec = agent.run("type hello in notepad and save")
    assert rec.outcome == "done"
    # Second monitor offset (left=1920) must be added to the click.
    assert ex.calls == [("launch", "notepad"), ("type", "hello"), ("click", 2450, 615)]
    assert (rec.dir / "report.html").exists() and len(list(rec.dir.glob("step_*.jpg"))) == 4
    assert "no visible change" in rec.steps[0].result  # identical frames


def test_risky_action_declined(tmp_path):
    agent, ex = make_agent(tmp_path, ['{"action": "click", "target": "Delete everything"}'], ask=lambda q: False)
    agent.parser = type("P", (), {"parse": lambda self, i: [Element(0, "text", (0, 0, 50, 20), "Delete everything")],
                                  "last_timing": {}})()
    rec = agent.run("x")
    assert rec.outcome == "aborted" and ex.calls == []


def test_invalid_replies_end_in_fail(tmp_path):
    agent, _ = make_agent(tmp_path, ["nonsense", "still nonsense"])
    assert agent.run("x").outcome == "fail"
