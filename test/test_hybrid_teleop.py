from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
import pytest

from rml_joy_teleop.hybrid_force import HybridForce
from rml_joy_teleop.teleop_workflow import TeleopWorkflow
from conftest import joy, pose


def arm(node):
    node.joy_cb(joy(7))
    node.joy_cb(joy())


def contact(node, clock):
    node.workflow.force.reset_contact()
    for _ in range(4):
        clock[0] += .01
        node.workflow.force.update(-7., clock[0])


def ack_stop(node):
    node.workflow.result("arm_stop", True)
    node.workflow.result("hybrid_stop", True)
    node.tick()


def test_force_rule_and_latched_fault():
    force = HybridForce()
    for i in range(4):
        force.update(-7., i * .01)
    assert force.contact
    assert force.filtered == pytest.approx(7 * (1 - .75 ** 4))
    force.filtered = 2.
    assert force.commanded_compression() == 9.5
    force.update(-100., 4)
    assert force.fault
    force.reset_contact()
    assert force.fault
    assert force.commanded_compression() == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_force(value):
    with pytest.raises(ValueError):
        HybridForce().update(value, 0)


def test_force_samples_must_be_distinct_and_consecutive():
    force = HybridForce()
    force.update(-8., 1.)
    force.update(-8., 1.01)
    assert force.count == 2
    force.update(-8., 1.01)
    assert force.count == 2 and not force.contact
    force.update(-8., 1.5)
    assert force.count == 1 and not force.contact
    force.update(-8., 1.51)
    force.update(-8., 1.52)
    assert force.contact


def test_position_mapping_and_rates_unchanged(node_factory):
    n, clock = node_factory()
    arm(n)
    n.joy_cb(joy(5, axes=(.5, 1., 1., 0., .3, 1., 0., 0.)))
    before = n.workflow.target()
    n.tick()
    after = n.workflow.target()
    np.testing.assert_allclose(after[:3] - before[:3], [.0002, .0001, .00006])
    assert sorted(t[0] for t in n.mock_timers) == [.01, .02]
    assert len(n.pub.messages) == 1
    assert not n.workflow.hybrid_pub.messages
    n.joy_cb(joy())
    n.tick()
    np.testing.assert_allclose(n.workflow.target(), after)
    assert len(n.pub.messages) == 2


def test_contact_without_rb_remains_armed(node_factory):
    n, clock = node_factory()
    arm(n)
    contact(n, clock)
    for _ in range(10):
        n.tick()
    assert n.armed and n.workflow.mode == "manual"
    assert not n.workflow.start_pub.messages
    assert not n.workflow.stop_pub.messages


def test_hybrid_xy_only_ack_gated_and_exclusive(node_factory):
    n, clock = node_factory()
    arm(n)
    contact(n, clock)
    n.joy_cb(joy(5, axes=(.5, 1., -1., 0., 1., 1., 1., 1.)))
    n.tick()
    w = n.workflow
    before = w.target()
    w.stream_hybrid()
    assert not w.hybrid_pub.messages and not n.pub.messages
    w.result("hybrid_start", True)
    clock[0] += .01
    n.tick()
    w.stream_hybrid()
    np.testing.assert_allclose(w.target()[2:], before[2:])
    assert w.target()[0] > before[0]
    msg = w.hybrid_pub.messages[-1]
    assert list(msg.control_mode) == [3, 3, 4, 0, 0, 0]
    assert msg.desired_force[2] == pytest.approx(-w.force.commanded_compression())
    assert msg.limit_vel[2] == pytest.approx(.002)
    assert not n.pub.messages


def test_rb_pause_resume_does_not_disarm_permanently(node_factory):
    n, clock = node_factory()
    arm(n)
    contact(n, clock)
    n.joy_cb(joy(5))
    n.tick()
    n.workflow.result("hybrid_start", True)
    n.joy_cb(joy())
    assert n.workflow.mode == "stopping"
    n.workflow.stream_hybrid()
    assert not n.workflow.hybrid_pub.messages
    ack_stop(n)
    assert n.armed and n.workflow.handoff_gate
    n.joy_cb(joy())  # neutral release after stop acknowledgment
    n.joy_cb(joy(5))
    n.tick()
    assert n.workflow.hybrid == "starting"
    assert n.workflow.force.contact


@pytest.mark.parametrize("which", ["pose", "force", "joy"])
def test_stale_feedback_stops(node_factory, which):
    n, clock = node_factory()
    arm(n)
    if which == "pose":
        n.workflow.pose_time -= 1
    elif which == "force":
        n.workflow.force.received -= 1
    else:
        n.workflow.joy_time -= 1
    n.tick()
    assert n.workflow.mode == "stopping"
    assert n.workflow.stop_pub.messages
    assert n.workflow.hybrid_stop_pub.messages
    assert not n.pub.messages


def test_late_start_requires_another_hybrid_stop(node_factory):
    n, clock = node_factory()
    arm(n)
    contact(n, clock)
    n.joy_cb(joy(5))
    n.tick()
    n.joy_cb(joy(1))
    w = n.workflow
    w.result("arm_stop", True)
    w.result("hybrid_stop", True)
    n.tick()
    assert w.mode == "stopping"
    w.result("hybrid_start", True)
    assert "hybrid_stop" in w.pending
    assert len(w.hybrid_stop_pub.messages) == 2
    n.tick()
    assert w.mode == "stopping"
    w.result("hybrid_stop", True)
    n.tick()
    assert w.mode == "idle"


@pytest.mark.parametrize("failed", [False, True])
def test_stop_missing_or_failed_ack_latches_restart(node_factory, failed):
    n, clock = node_factory()
    arm(n)
    n.joy_cb(joy(1))
    if failed:
        n.workflow.result("arm_stop", False)
        n.workflow.result("hybrid_stop", True)
    clock[0] += 2.1
    n.tick()
    assert n.workflow.mode == "fault"
    n.workflow.result("arm_stop", True)
    n.workflow.result("hybrid_stop", True)
    n.joy_cb(joy(7))
    n.tick()
    assert n.workflow.mode == "fault" and not n.armed


def test_b_priority_over_start_and_malformed_axes(node_factory):
    n, clock = node_factory()
    arm(n)
    n.joy_cb(joy(1, 5, 7, axes=[]))
    assert n.workflow.mode == "stopping"
    assert not n.pub.messages


def test_gripper_and_home_legacy_but_home_disabled_in_contact(node_factory):
    n, clock = node_factory()
    n.joy_cb(joy(3))
    assert n.gripper_bridge_pub.messages[-1].position == 1000
    n.joy_cb(joy())
    arm(n)
    n.joy_cb(joy(0))
    assert n.gripper_bridge_pub.messages[-1].position == 0
    n.joy_cb(joy(8))
    assert n.tx == n.home_tx
    contact(n, clock)
    n.tx = .4
    n.joy_cb(joy(8))
    assert n.tx == .4


def test_force_fault_survives_handoff_reset(node_factory):
    n, clock = node_factory()
    n.workflow.force.update(-100., clock[0] + .001)
    n.workflow.force.reset_contact()
    n.joy_cb(joy(7))
    assert n.workflow.mode == "idle" and not n.armed


def test_start_edge_not_reused_after_stop(node_factory):
    n, clock = node_factory()
    arm(n)
    n.joy_cb(joy(1, 7))
    ack_stop(n)
    n.joy_cb(joy(7))
    assert n.workflow.mode == "idle"
    n.joy_cb(joy())
    n.joy_cb(joy(7))
    assert n.workflow.mode == "manual"


def test_start_timeout(node_factory):
    n, clock = node_factory()
    arm(n)
    contact(n, clock)
    n.joy_cb(joy(5))
    n.tick()
    clock[0] += 2.1
    n.workflow.pose_time = n.workflow.joy_time = n.workflow.force.received = clock[0]
    n.tick()
    assert n.workflow.mode == "stopping"
    assert not n.pub.messages


def test_gripper_goal_accepted_after_cancel_is_cancelled(node_factory):
    n, clock = node_factory(auto_pick_on_start=True)
    w = n.workflow
    n.joy_cb(joy(7))
    n.pose_cb(pose())
    n.tick()
    assert "gripper" in w.pending
    n.joy_cb(joy(1))
    result, cancel = Future(), Future()
    goal = SimpleNamespace(accepted=True, get_result_async=lambda: result,
                           cancel_goal_async=lambda: cancel)
    w.gripper_client.future.set_result(goal)
    assert "gripper_cancel" in w.pending
    cancel.set_result(SimpleNamespace(return_code=0))
    result.set_result(SimpleNamespace(status=5, result=SimpleNamespace(error_code=0)))
    ack_stop(n)
    assert w.mode == "idle"
    assert not w.motion_pubs["movej"].messages


def test_recording_requires_pickup(node_factory):
    with pytest.raises(ValueError, match="requires"):
        node_factory(record_demonstrations=True)


@pytest.mark.parametrize("kind", ["axes", "buttons"])
def test_out_of_range_joystick_requests_stop(node_factory, kind):
    n, clock = node_factory()
    arm(n)
    msg = joy(5)
    if kind == "axes":
        msg.axes[0] = 2.
    else:
        msg.buttons[0] = -1
    n.joy_cb(msg)
    assert n.workflow.mode == "stopping"
    assert not n.pub.messages


def test_hybrid_stale_target_prevents_stream(node_factory):
    n, clock = node_factory()
    arm(n)
    contact(n, clock)
    n.joy_cb(joy(5))
    n.tick()
    w = n.workflow
    w.result("hybrid_start", True)
    w.target_updated -= .3
    w.stream_hybrid()
    assert w.mode == "stopping" and not w.hybrid_pub.messages
