"""End-to-end workflow using only the mock node and real local file writer."""
import json
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image
from std_msgs.msg import UInt16

from conftest import joy, pose
from test_demonstration import fresh
from test_hybrid_teleop import ack_stop, contact


def feedback(n, clock, dt=.01):
    clock[0] += dt
    n.pose_cb(pose())
    fresh(n.workflow.sensors, clock[0])
    n.workflow.force.update(0., clock[0])
    n.workflow.joy_time = clock[0]


def handoff(node_factory):
    n, clock = node_factory(auto_pick_on_start=True, record_demonstrations=True)
    w = n.workflow
    w.mode = "handoff"
    w.handoff_gate = True
    fresh(w.sensors, clock[0])
    n.tick()
    assert w.recorder.writer.ready.wait(5)
    return n, clock


def recording(node_factory):
    n, clock = handoff(node_factory)
    n.joy_cb(joy())
    n.joy_cb(joy(5))
    assert n.workflow.recorder.active
    return n, clock


def finalize(n, clock):
    w = n.workflow
    writer = w.recorder.writer
    w.result("arm_stop", True)
    w.result("hybrid_stop", True)
    n.tick()
    # A missing terminal frame gets at most 250 ms to arrive.
    if not writer.finish_event.is_set():
        clock[0] += .26
        n.tick()
    writer.thread.join(5)
    assert writer.done.is_set()
    n.tick()
    return writer


def read_outcome(writer):
    path = writer.final_path
    return json.loads((path if path.suffix == ".json" else path / "outcome.json").read_text())


def test_initial_neutral_rb_gate_and_jump_free_handoff(node_factory):
    n, clock = handoff(node_factory)
    w = n.workflow
    n.tx = .5
    n.vx = 1
    n.joy_cb(joy(5))
    n.tick()
    assert not n.pub.messages and not w.recorder.started
    n.joy_cb(joy(axes=(1., 0., 1., 0., 0., 1., 0., 0.)))
    n.joy_cb(joy(5))
    assert not w.recorder.started
    n.joy_cb(joy())
    n.joy_cb(joy(5))
    n.tick()
    assert n.armed and w.recorder.started and w.recorder.index == 1
    assert n.vx == n.vy == n.vz == 0
    assert n.pub.messages[-1].pose.position.x == pytest.approx(.430)
    assert w.recorder.last_target[0] == pytest.approx(.430)


def test_missing_image_blocks_handoff_without_motion(node_factory):
    n, clock = handoff(node_factory)
    del n.workflow.sensors.cache["image"]
    n.joy_cb(joy())
    n.joy_cb(joy(5))
    n.tick()
    assert n.workflow.mode == "handoff"
    assert not n.workflow.recorder.started and not n.pub.messages


def test_15hz_recording_does_not_change_100hz_position_stream(node_factory):
    n, clock = recording(node_factory)
    for _ in range(101):
        feedback(n, clock, .01)
        n.tick()
    assert len(n.pub.messages) == 101
    assert n.workflow.recorder.index == 16  # Initial frame plus 15 scheduled frames.
    assert not n.workflow.hybrid_pub.messages


def test_15hz_recording_does_not_change_50hz_hybrid_stream(node_factory):
    n, clock = recording(node_factory)
    w = n.workflow
    contact(n, clock)
    n.tick()
    w.result("hybrid_start", True)
    initial = w.recorder.index
    for index in range(100):
        feedback(n, clock, .01)
        n.tick()
        if index % 2 == 0:
            w.stream_hybrid()
    assert len(w.hybrid_pub.messages) == 50
    assert w.recorder.index - initial == 15
    assert not n.pub.messages


@pytest.mark.parametrize("buttons,expected", [
    ((0,), "success"), ((2,), "failure"), ((1,), "discard"),
    ((0, 2), "failure"), ((0, 1, 2), "discard"),
])
def test_outcome_priority_latch_and_no_gripper(node_factory, buttons, expected):
    n, clock = recording(node_factory)
    feedback(n, clock)
    n.joy_cb(joy(*buttons, 5))
    w = n.workflow
    assert w.mode == "stopping" and w.outcome["requested"] == expected
    n.joy_cb(joy(0, 1, 2, 5))
    n.tick()
    assert w.outcome["requested"] == expected
    assert not n.pub.messages and not w.hybrid_pub.messages
    assert not n.gripper_bridge_pub.messages
    writer = finalize(n, clock)
    assert read_outcome(writer)["status"] == expected
    assert not n.armed
    n.joy_cb(joy(0))
    assert not n.gripper_bridge_pub.messages
    n.joy_cb(joy())
    n.joy_cb(joy(0))
    assert len(n.gripper_bridge_pub.messages) == 1


def test_recording_disables_home_and_y(node_factory):
    n, clock = recording(node_factory)
    target = n.workflow.target()
    n.joy_cb(joy(3, 5, 8))
    np.testing.assert_allclose(n.workflow.target(), target)
    assert not n.gripper_bridge_pub.messages
    assert n.workflow.recorder.active


def test_resume_has_new_segment_and_no_reseeding_action(node_factory):
    n, clock = recording(node_factory)
    w = n.workflow
    feedback(n, clock, .08)
    n.tick()
    n.joy_cb(joy())
    assert w.mode == "stopping" and not w.recorder.active
    ack_stop(n)
    assert w.handoff_gate and n.armed
    feedback(n, clock)
    n.joy_cb(joy(5))
    assert not w.recorder.active
    n.joy_cb(joy())
    n.joy_cb(joy(5))
    assert w.recorder.active and w.recorder.segment == 1
    assert w.recorder.xy.tolist() == [0., 0.]


@pytest.mark.parametrize("button,status", [(0, "success"), (2, "failure"), (1, "discard")])
def test_outcome_during_pause_stop_ack_wait(node_factory, button, status):
    n, clock = recording(node_factory)
    n.joy_cb(joy())
    assert n.workflow.mode == "stopping"
    n.joy_cb(joy(button))
    assert n.workflow.outcome["outcome_while_paused"]
    writer = finalize(n, clock)
    outcome = read_outcome(writer)
    assert outcome["status"] == status and not outcome["terminal_observation"]
    assert n.workflow.mode == "idle" and not n.armed


def test_terminal_waits_for_new_image_without_commands(node_factory):
    n, clock = recording(node_factory)
    w = n.workflow
    n.joy_cb(joy(0))
    ack_stop(n)
    assert w.mode == "stopping" and not w.recorder.terminal
    feedback(n, clock, .03)
    writer = finalize(n, clock)
    assert read_outcome(writer)["terminal_observation"]
    assert read_outcome(writer)["status"] == "success"
    assert not n.pub.messages


def test_missing_terminal_frame_is_incomplete(node_factory):
    n, clock = recording(node_factory)
    n.joy_cb(joy(0))
    writer = finalize(n, clock)
    assert read_outcome(writer)["status"] == "incomplete"
    assert read_outcome(writer)["requested"] == "success"


def test_paused_label_requires_fresh_feedback_without_terminal_sample(node_factory):
    n, clock = recording(node_factory)
    n.joy_cb(joy())
    n.workflow.sensors.cache["gripper"]["receipt_monotonic"] -= .3
    n.joy_cb(joy(0))
    writer = finalize(n, clock)
    outcome = read_outcome(writer)
    assert outcome["status"] == "incomplete"
    assert outcome["outcome_while_paused"] and not outcome["terminal_observation"]


@pytest.mark.parametrize("started", [False, True])
def test_deployed_calibration_change_stops_recording(node_factory, started):
    n, clock = (recording if started else handoff)(node_factory)
    message = SimpleNamespace(
        node="/realman_gripper_bridge",
        new_parameters=[], deleted_parameters=[],
        changed_parameters=[SimpleNamespace(name="hkv_feedback_open_register")])
    n.workflow.parameter_cb(message)
    assert n.workflow.mode == "stopping"
    assert "calibration changed" in n.workflow.outcome["reason"]


def test_calibration_readback_change_invalidates_prepared_metadata(node_factory):
    n, clock = handoff(node_factory)
    from rcl_interfaces.msg import ParameterValue, ParameterType
    future = Future()
    future.set_result(SimpleNamespace(values=[
        ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=1000),
        ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=0),
        ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=.2),
    ]))
    n.workflow.calibration_future = future
    n.tick()
    assert n.workflow.mode == "stopping" and not n.pub.messages
    assert "calibration changed" in n.workflow.outcome["reason"]


@pytest.mark.parametrize("key", ["image", "gripper"])
def test_stale_recording_feedback_stops(node_factory, key):
    n, clock = recording(node_factory)
    n.workflow.sensors.cache[key]["receipt_monotonic"] -= .3
    n.tick()
    assert n.workflow.mode == "stopping"
    assert not n.pub.messages
    writer = finalize(n, clock)
    assert read_outcome(writer)["status"] == "incomplete"


def test_writer_failure_requests_stop_and_blocks_restart(node_factory):
    n, clock = recording(node_factory)
    w = n.workflow
    w.recorder.writer.error = "injected disk failure"
    n.tick()
    assert w.mode == "stopping" and w.stop_pub.messages
    writer = finalize(n, clock)
    assert w.mode == "fault"
    n.joy_cb(joy())
    n.joy_cb(joy(7))
    assert w.mode == "fault" and not n.armed


def test_queue_failure_does_not_override_b_discard(node_factory, monkeypatch):
    n, clock = recording(node_factory)
    def overflow(*args, **kwargs):
        raise RuntimeError("Recorder queue overflow")
    monkeypatch.setattr(n.workflow.recorder.writer, "submit", overflow)
    n.joy_cb(joy(0, 1, 2))
    assert n.workflow.outcome["requested"] == "discard"
    assert n.workflow.mode == "stopping"


def test_start_blocked_until_finalization_and_new_edge(node_factory):
    n, clock = recording(node_factory)
    feedback(n, clock)
    n.joy_cb(joy(0))
    n.joy_cb(joy(7))
    writer = finalize(n, clock)
    assert read_outcome(writer)["status"] == "success"
    n.joy_cb(joy(7))
    assert n.workflow.mode == "idle"
    n.joy_cb(joy())
    n.joy_cb(joy(7))
    assert n.workflow.mode == "pick"


def test_missing_stop_ack_invalidates_success_and_restart(node_factory):
    n, clock = recording(node_factory)
    feedback(n, clock)
    n.joy_cb(joy(0))
    writer = n.workflow.recorder.writer
    clock[0] += 2.1
    n.tick()
    writer.thread.join(5)
    n.tick()
    assert read_outcome(writer)["status"] == "incomplete"
    assert not read_outcome(writer)["stop_confirmed"]
    assert n.workflow.mode == "fault"
    n.workflow.result("arm_stop", True)
    n.workflow.result("hybrid_stop", True)
    n.joy_cb(joy(7))
    assert n.workflow.mode == "fault"


def test_full_command_fields_recorded_and_contact_reference_preserved(node_factory):
    n, clock = recording(node_factory)
    w = n.workflow
    n.tz = .193
    n.tick()
    contact(n, clock)
    n.tick()
    w.result("hybrid_start", True)
    clock[0] += .01
    n.tick()
    w.stream_hybrid()
    assert n.tz == .193
    assert len(n.pub.messages) == 1
    feedback(n, clock)
    n.joy_cb(joy(0))
    writer = finalize(n, clock)
    events = [json.loads(line) for line in (writer.final_path / "events.jsonl").read_text().splitlines()]
    commands = [event for event in events if event["type"] == "command"]
    assert "follow" in commands[0]["published"]
    hybrid = commands[-1]["published"]
    assert hybrid["control_mode"] == [3, 3, 4, 0, 0, 0]
    assert hybrid["limit_vel"][2] == pytest.approx(.002)
    assert hybrid["pose"]["position"]["z"] == .193


def test_coordinate_and_camera_layout_changes_stop(node_factory):
    n, clock = recording(node_factory)
    n.workflow.coordinate_cb(UInt16(data=2))
    assert n.workflow.mode == "stopping"
    assert "coordinate" in n.workflow.outcome["reason"]


@pytest.mark.parametrize("started", [False, True])
def test_camera_frame_change_stops(node_factory, started):
    n, clock = (recording if started else handoff)(node_factory)
    w = n.workflow
    w.image_layout = (1, 1, "rgb8", "original")
    image = Image(width=1, height=1, encoding="rgb8", step=3, data=[0, 0, 0])
    image.header.frame_id = "changed"
    w.image_cb(image)
    assert w.mode == "stopping" and "camera" in w.outcome["reason"]


def test_full_mock_pickup_to_handoff(node_factory):
    n, clock = node_factory(auto_pick_on_start=True)
    w = n.workflow
    n.joy_cb(joy(7))
    feedback(n, clock)
    n.tick()

    def gripper_done():
        result = Future()
        goal = SimpleNamespace(accepted=True, get_result_async=lambda: result)
        w.gripper_client.future.set_result(goal)
        result.set_result(SimpleNamespace(status=4, result=SimpleNamespace(error_code=0)))

    gripper_done()
    assert w.motion_pubs["movej"].messages[-1].speed == 10
    w.result("movej", True)
    w.result("movel", True)
    from rml_joy_teleop.auto_pick import GripFeedback
    w.grip = GripFeedback(1, clock[0], 1000, (0,) * 6)
    n.tick()
    gripper_done()
    for seq in (2, 3):
        feedback(n, clock)
        w.grip = GripFeedback(seq, clock[0], 860, (0,) * 6)
        n.tick()
    feedback(n, clock, 1.51)
    n.tick()
    assert w.picker.stage == "lift"
    w.result("movel", True)
    w.result("movel", True)
    w.result("movel", True)
    assert w.picker.stage == "verify"
    n.tick()
    assert w.mode == "pick"
    feedback(n, clock)
    n.tick()
    assert w.mode == "handoff"
    assert not n.pub.messages and not w.hybrid_pub.messages
    n.joy_cb(joy())
    n.joy_cb(joy(5))
    n.tick()
    assert w.mode == "manual" and n.armed
    assert n.pub.messages[-1].pose.position.x == pytest.approx(.43)
