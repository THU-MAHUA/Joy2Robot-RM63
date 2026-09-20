import json
import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import pose
from rml_joy_teleop import demonstration as demo
from rml_joy_teleop.demonstration import (
    ObservationBuffer, EpisodeWriter, Recorder, Scheduler, rgb_image,
    recover_staging, STATE_COMPONENTS,
)
from rml_joy_teleop.demo_loader import load_transitions


PAYLOAD = (2, 1, "rgb8", 6, bytes([250, 10, 0, 250, 10, 0]))


def fresh(buffer, now=1., include_gripper=True):
    buffer.pose(pose(), now - .01, int((now - .01) * 1e9))
    buffer.pose(pose(), now, int(now * 1e9))
    for key, value in (("image", PAYLOAD), ("wrench", [1., 2., -3., 4., 5., 6.]),
                       ("coordinate", 1)):
        buffer.put(key, value, now, int(now * 1e9), 42 if key == "image" else None)
    if include_gripper:
        buffer.put("gripper", 841, now, int(now * 1e9), None)
        buffer.set_calibration(1000, 0, .1)
    return buffer


def wait_writer(writer):
    writer.thread.join(5)
    assert writer.done.is_set()
    assert not writer.error


def test_measured_state_and_calibration():
    buffer = fresh(ObservationBuffer())
    snapshot = buffer.snapshot(1.)
    state = snapshot["state"]
    assert sum(map(len, state.values())) == 19
    assert len(STATE_COMPONENTS) == 19
    np.testing.assert_allclose(state["tcp_pose"][:3], [.43, -.01, .03125])
    assert state["gripper_pose"] == pytest.approx([.0159])
    assert state["tcp_force"] == [1., 2., -3.]
    assert state["tcp_torque"] == [4., 5., 6.]
    assert snapshot["raw_gripper_register"] == 841
    assert snapshot["wrench_coordinate"] == 1
    assert snapshot["timing"]["image"]["source_ns"] == 42
    buffer.set_calibration(100, 0, .1)
    with pytest.raises(ValueError, match="range"):
        buffer.snapshot(1.)


def test_optional_gripper_for_online_callers():
    buffer = fresh(ObservationBuffer(), include_gripper=False)
    snapshot = buffer.snapshot(1., None, False)
    assert sum(map(len, snapshot["state"].values())) == 18
    assert "raw_gripper_register" not in snapshot
    with pytest.raises(ValueError, match="gripper"):
        buffer.snapshot(1.)
    buffer.put("gripper", 841, 0, 0)
    assert buffer.snapshot(1., None, False)


@pytest.mark.parametrize("key", ["pose", "wrench", "image", "coordinate", "gripper"])
def test_stale_and_missing_feedback(key):
    buffer = fresh(ObservationBuffer())
    buffer.cache[key]["receipt_monotonic"] -= .251
    with pytest.raises(ValueError, match=key):
        buffer.snapshot(1.)
    del buffer.cache[key]
    with pytest.raises(ValueError, match=key):
        buffer.snapshot(1.)


def test_velocity_quaternion_wrap_and_gap():
    buffer = ObservationBuffer()
    q1 = Rotation.from_euler("z", 179, degrees=True).as_quat()
    q2 = Rotation.from_euler("z", -179, degrees=True).as_quat()
    buffer.pose(pose(q=tuple(q1)), 1., 1)
    buffer.pose(pose(x=.431, q=tuple(q2)), 1.1, 2)
    assert buffer.velocity[0] == pytest.approx(.01)
    assert buffer.velocity[5] == pytest.approx(np.deg2rad(2) / .1)
    buffer.pose(pose(q=tuple(-q2)), 1.2, 3)
    assert buffer.velocity[5] == pytest.approx(0)
    buffer.pose(pose(), 2., 4)
    assert buffer.velocity is None
    buffer.pose(pose(), 2.01, 5)
    assert buffer.velocity is not None


@pytest.mark.parametrize("encoding,data,step", [
    ("rgb8", [250, 10, 0], 3), ("bgr8", [0, 10, 250], 3),
    ("rgba8", [250, 10, 0, 50], 4), ("bgra8", [0, 10, 250, 50], 4),
    ("bgr8", [0, 10, 250, 99, 99], 5),
])
def test_rgb_encoding_and_stride(encoding, data, step):
    rgb = rgb_image((1, 1, encoding, step, bytes(data)))
    assert rgb.shape == (128, 128, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [250, 10, 0]


def test_full_frame_area_resize():
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    data[:, 128:, 0] = 200
    rgb = rgb_image((256, 256, "rgb8", 256 * 3, data.tobytes()))
    assert rgb[0, 0, 0] == 0 and rgb[0, -1, 0] == 200


def test_scheduler_15hz_no_catchup_bursts():
    scheduler = Scheduler()
    scheduler.start(0)
    assert not scheduler.due(.066)
    assert scheduler.due(1 / 15)
    assert not scheduler.due(1 / 15)
    assert scheduler.due(.5)
    assert not scheduler.due(.5)
    assert scheduler.next > .5


def test_frames_not_duplicated():
    buffer = fresh(ObservationBuffer())
    sequence = buffer.snapshot(1.)["timing"]["image"]["sequence"]
    assert buffer.snapshot(1.01, sequence) is None
    with pytest.raises(ValueError):
        buffer.snapshot(1.3, sequence)


def make_recorder(tmp_path):
    buffer = fresh(ObservationBuffer())
    recorder = Recorder(tmp_path, {"record_rate_hz": 15.}, buffer)
    assert recorder.writer.ready.wait(5)
    return recorder, buffer


def test_command_accumulation_reseed_and_segments(tmp_path):
    recorder, buffer = make_recorder(tmp_path)
    assert recorder.resume(1.)
    target = np.array([.43, -.01, .19, 1, 0, 0, 0])
    recorder.seed(target)
    recorder.command(target + [.001, 0, 0, 0, 0, 0, 0], "position", 1.01)
    recorder.command(target + [.002, .003, 0, 0, 0, 0, 0], "position", 1.02)
    fresh(buffer, 1.08)
    recorder.tick(1.08)
    recorder.pause(1.09)
    recorder.seed(target + [.01, .01, 0, 0, 0, 0, 0])
    fresh(buffer, 2.)
    assert recorder.resume(2.)
    fresh(buffer, 2.08)
    recorder.capture(2.08, terminal=True)
    recorder.writer.finish(dict(status="success", stop_confirmed=True))
    wait_writer(recorder.writer)
    frames = [json.loads(line) for line in (recorder.writer.final_path / "frames.jsonl").read_text().splitlines()]
    assert frames[1]["action_xy_m"] == pytest.approx([.002, .003])
    assert frames[1]["dt"] == pytest.approx(.08)
    assert frames[2]["initial"] and frames[2]["action_xy_m"] is None
    transitions = list(load_transitions(recorder.writer.final_path))
    assert len(transitions) == 2
    assert transitions[0]["truncated"] and transitions[0]["rewards"] == 0
    assert transitions[1]["terminated"] and transitions[1]["rewards"] == 1
    assert transitions[1]["observations"]["images"]["wrist_1"][0, 0].tolist() == [250, 10, 0]


def test_z_and_rotation_invalidate_xy_interval(tmp_path):
    recorder, buffer = make_recorder(tmp_path)
    recorder.resume(1.)
    target = np.array([.43, -.01, .19, 1, 0, 0, 0])
    recorder.seed(target)
    recorder.command(target + [0, 0, .001, 0, 0, 0, 0], "position", 1.01)
    fresh(buffer, 1.1)
    recorder.capture(1.1)
    assert recorder.compatible
    recorder.command(target + [0, 0, .001, 0, 0, 0, 0], "hybrid", 1.11)
    fresh(buffer, 1.2)
    recorder.capture(1.2, terminal=True)
    recorder.writer.finish(dict(status="success", stop_confirmed=True))
    wait_writer(recorder.writer)
    transitions = list(load_transitions(recorder.writer.final_path))
    assert len(transitions) == 1


def test_discard_deletes_only_current_attempt(tmp_path):
    existing = tmp_path / "success" / "old"
    existing.mkdir(parents=True)
    recorder, buffer = make_recorder(tmp_path)
    recorder.resume(1.)
    recorder.writer.finish(dict(status="discard", requested="discard"))
    wait_writer(recorder.writer)
    assert existing.is_dir()
    assert not recorder.writer.path.exists()
    assert recorder.writer.final_path.suffix == ".json"
    assert recorder.writer.final_path.parent.name == "cancelled"


def test_writer_failure_preserves_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.cv2, "imwrite", lambda *a: False)
    recorder, buffer = make_recorder(tmp_path)
    recorder.resume(1.)
    recorder.writer.thread.join(5)
    assert recorder.writer.error
    assert recorder.writer.path.is_dir()
    recovered = recover_staging(tmp_path)
    assert len(recovered) == 1
    assert json.loads((tmp_path / "incomplete" / recorder.writer.identifier / "outcome.json").read_text())["status"] == "incomplete"


def test_bounded_queue_never_blocks_callback():
    writer = EpisodeWriter.__new__(EpisodeWriter)
    writer.error = None
    writer.done, writer.finish_event = threading.Event(), threading.Event()
    writer.queue = queue.Queue(1)
    writer.submit("event", {})
    with pytest.raises(RuntimeError, match="overflow"):
        writer.submit("event", {})


def test_loader_no_terminal_reward_for_paused_outcome(tmp_path):
    recorder, buffer = make_recorder(tmp_path)
    recorder.resume(1.)
    fresh(buffer, 1.1)
    recorder.capture(1.1)
    recorder.pause(1.11)
    recorder.writer.finish(dict(status="success", stop_confirmed=True, outcome_while_paused=True))
    wait_writer(recorder.writer)
    transitions = list(load_transitions(recorder.writer.final_path))
    assert transitions[-1]["truncated"]
    assert transitions[-1]["rewards"] == 0
