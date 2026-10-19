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

    def parse(self, image, icons=True):
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
    s = Settings(runs_dir=tmp_path, action_pause=0, max_steps=6, trajectories_dir=tmp_path / "traj")
    frames = iter(lambda: screen.Frame(Image.new("RGB", (1920, 1080), "white"), left=1920, top=0), None)
    ex = RecordingExecutor()
    agent = Agent(s, model=ScriptedModel(replies), parser=FakeParser(), executor=ex, capture=lambda: next(frames),
                  ask=ask)
    agent.title = lambda: "Untitled - Notepad"
    return agent, ex


def test_full_loop(tmp_path):
    agent, ex = make_agent(tmp_path, [
        '{"steps": ["open notepad", "type \\"hello\\"", "click Save"]}',  # decomposition
        '{"done": true, "reason": "text typed and saved"}',  # final visual check
    ])
    rec = agent.run("type hello in notepad and save")
    assert rec.outcome == "done", rec.summary
    # Every plan step ran as a direct skill (no planner calls); the click matched OCR text "Save",
    # and the second monitor's offset (left=1920) was added to it.
    # Opening an editor and then typing inserts ctrl+n so the text lands in a fresh document.
    assert ex.calls == [("launch", "notepad"), ("hotkey", ("ctrl", "n")), ("type", "hello"), ("click", 2450, 615)]
    assert [s.source for s in rec.steps[:5]] == ["skill"] * 5
    assert (rec.dir / "report.html").exists() and len(list(rec.dir.glob("step_*.jpg"))) == 6
    assert "no visible change" in rec.steps[0].result  # identical frames


def test_planner_handles_open_ended_steps(tmp_path):
    agent, ex = make_agent(tmp_path, [
        '{"steps": ["make the text bigger"]}',
        '{"thought": "zoom", "action": "hotkey", "keys": ["ctrl", "plus"]}',
        '{"action": "done", "summary": "bigger"}',
        '{"done": true, "reason": "ok"}',
    ])
    rec = agent.run("make the text bigger")
    assert rec.outcome == "done" and ex.calls == [("hotkey", ("ctrl", "plus"))]
    assert rec.steps[0].source == "planner"


def test_failed_final_check_falls_back_to_planning(tmp_path):
    agent, ex = make_agent(tmp_path, [
        '{"steps": ["press enter"]}',
        '{"done": false, "reason": "nothing happened"}',
        '{"action": "fail", "summary": "cannot"}',
    ])
    rec = agent.run("x")
    assert rec.outcome == "fail" and ex.calls == [("hotkey", ("enter",))]


def test_risky_action_declined(tmp_path):
    agent, ex = make_agent(tmp_path, ['{"steps": ["click Delete everything"]}'], ask=lambda q: False)
    delete = [Element(0, "text", (0, 0, 50, 20), "Delete everything")]
    agent.parser = type("P", (), {"parse": lambda self, i, icons=True: delete, "last_timing": {}})()
    rec = agent.run("x")
    assert rec.outcome == "aborted" and ex.calls == []


def test_invalid_replies_end_in_fail(tmp_path):
    agent, _ = make_agent(tmp_path, ['{"steps": ["do it"]}', "nonsense", "still nonsense"])
    assert agent.run("x").outcome == "fail"


def test_adaptive_vision_policy(tmp_path):
    from omni_agent.actions import Action
    from omni_agent.planner import StepRecord

    agent, _ = make_agent(tmp_path, [])
    many = [Element(i, "text", (0, i * 20, 50, i * 20 + 15), f"t{i}") for i in range(20)]
    few = many[:3]
    ok = StepRecord(Action(action="click", target="x"), "clicked 'x'")
    stuck = StepRecord(Action(action="click", target="x"), "clicked 'x' (no visible change)")
    assert agent._needs_image(many, []) is False
    assert agent._needs_image(few, []) is True
    assert agent._needs_image(many, [ok]) is False
    assert agent._needs_image(many, [stuck]) is True
    agent.s.vision = "always"
    assert agent._needs_image(many, [ok]) is True


def test_refuses_to_type_into_existing_file(tmp_path):
    agent, ex = make_agent(tmp_path, ['{"steps": ["type \\"hi\\""]}'], ask=lambda q: False)
    agent.title = lambda: "recovery-codes.txt - Notepad"
    rec = agent.run("write hi")
    assert rec.outcome == "aborted" and "recovery-codes.txt" in rec.summary and ex.calls == []


def test_editor_plan_starts_fresh_document(tmp_path):
    agent, ex = make_agent(tmp_path, ['{"steps": ["open Notepad", "type \\"hi\\""]}', '{"done": true}'])
    agent.run("open notepad and write hi")
    assert ex.calls == [("launch", "Notepad"), ("hotkey", ("ctrl", "n")), ("type", "hi")]


def test_replay_skips_the_model_and_hands_over_on_lost_anchor(tmp_path):
    goal = "type hello in notepad and save"
    agent, ex = make_agent(tmp_path, [
        '{"steps": ["open notepad", "type \\"hello\\"", "click Save"]}',
        '{"done": true, "reason": "ok"}',
    ])
    assert agent.run(goal).outcome == "done"
    first_calls = list(ex.calls)

    # Same goal again: replayed from the stored trajectory. Only the final check calls the model.
    agent2, ex2 = make_agent(tmp_path, ['{"done": true, "reason": "ok"}'])
    rec = agent2.run(goal)
    assert rec.outcome == "done" and ex2.calls == first_calls
    # launch, ctrl+n, wait, type, click: all replayed
    assert [st.source for st in rec.steps[:5]] == ["replay"] * 5
    assert "(text)" in rec.steps[4].result  # the Save click was re-found by its OCR text

    # The "Save" text is gone and the stored image patch is blank: replay stops at that plan step
    # and the agent takes over there, without re-typing what was already typed.
    agent3, ex3 = make_agent(tmp_path, ['{"x": -1, "y": -1}', '{"action": "fail", "summary": "no Save button"}'])
    agent3.parser = type("P", (), {"parse": lambda self, i, icons=True: [], "last_timing": {}})()
    rec = agent3.run(goal)
    assert rec.outcome == "fail"
    assert ex3.calls == first_calls[:3]  # launch, ctrl+n, type - each exactly once
