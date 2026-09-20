from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rml_joy_teleop.auto_pick import Picker, GripFeedback, waypoints, tcp_position, PREINSERT


class Adapter:
    def __init__(self):
        self.commands = []
        self.errors = []
        self.pose_sequence = 10

    def motion(self, kind, p, q):
        self.commands.append((kind, np.array(p), np.array(q)))

    def gripper(self, position, duration):
        self.commands.append(("gripper", position, duration))

    def fail(self, reason):
        self.errors.append(reason)
        self.picker.cancel()


@pytest.fixture
def picker():
    adapter = Adapter()
    p = Picker(adapter)
    adapter.picker = p
    return p, adapter


FLANGE = np.array([.43, -.01, .19, 1., 0., 0., 0.])


def advance_to_grip(p):
    p.start(0, 0)
    p.service(.01, FLANGE, 1, .01, None)
    p.result(True, .02)  # open
    p.result(True, .03)  # approach
    p.result(True, .04)  # grab
    p.service(.05, FLANGE, 2, .05, GripFeedback(1, .05, 1000, (0,) * 6))
    p.result(True, .06)  # close


def test_rotated_offset_and_geometry():
    points, q = waypoints(FLANGE)
    np.testing.assert_allclose(points["grab"], [.281, -.007711, .158])
    np.testing.assert_allclose(points["approach"], [.281, -.007711, .188])
    np.testing.assert_allclose(tcp_position(np.r_[points["preinsert"], q]), PREINSERT)
    q2 = Rotation.from_euler("xyz", [3., .1, .2]).as_quat()
    points, q = waypoints(np.r_[FLANGE[:3], q2])
    np.testing.assert_allclose(tcp_position(np.r_[points["preinsert"], q]), PREINSERT)


@pytest.mark.parametrize("flange,grab", [
    (FLANGE, [10000, 0, 0]), (FLANGE, [float("nan"), 0, 0]),
    (np.r_[FLANGE[:3], [0, 0, 0, 1]], [281, -7, 158]),
])
def test_invalid_waypoints(flange, grab):
    with pytest.raises(ValueError):
        waypoints(flange, grab)


def test_sequence_and_fresh_preinsert_verification(picker):
    p, a = picker
    advance_to_grip(p)
    assert [c[0] for c in a.commands] == ["gripper", "movej", "movel", "gripper"]
    assert a.commands[0][1] == .1 and a.commands[3][1] == .014
    for seq in (2, 3):
        p.service(seq / 10, FLANGE, 3, seq / 10, GripFeedback(seq, seq / 10, 860, (0,) * 6))
    assert p.stage == "settle"
    p.service(1.79, FLANGE, 3, 1.79, None)
    assert p.stage == "settle"
    p.service(1.81, FLANGE, 3, 1.81, None)
    assert p.stage == "lift"
    p.result(True, 1.82)
    p.result(True, 1.83)
    p.result(True, 1.84)
    assert p.stage == "verify"
    measured = np.r_[p.points["preinsert"], p.orientation]
    p.service(1.85, measured, 10, 1.85, None)
    assert p.stage == "verify"
    p.service(1.86, measured, 11, 1.86, None)
    assert p.stage == "done"
    assert not a.errors


def test_tactile_confirmation_two_distinct_fresh_samples(picker):
    p, a = picker
    advance_to_grip(p)
    feedback = GripFeedback(2, .1, 1000, (21, 0, 0, 0, 0, 0))
    p.service(.1, FLANGE, 2, .1, feedback)
    p.service(.11, FLANGE, 2, .11, feedback)
    assert p.stage == "grip"
    p.service(.12, FLANGE, 2, .12, GripFeedback(3, .12, 1000, (21, 0, 0, 0, 0, 0)))
    assert p.stage == "settle"


@pytest.mark.parametrize("stage", [
    "fresh_pose", "open", "approach", "grab", "baseline", "close",
    "grip", "settle", "lift", "transfer", "preinsert", "verify",
])
def test_cancellation_never_advances_on_late_results(picker, stage):
    p, a = picker
    p.stage = stage
    p.cancel()
    p.result(True, 0)
    p.service(100, FLANGE, 100, 100, None)
    assert not a.commands and p.stage == "cancelled"


@pytest.mark.parametrize("stage", [
    "fresh_pose", "open", "approach", "grab", "baseline", "close", "grip", "lift", "transfer", "preinsert", "verify",
])
def test_timeouts(picker, stage):
    p, a = picker
    p.stage, p.deadline = stage, 1
    p.service(2, FLANGE, 100, 2, None)
    assert a.errors and not a.commands


def test_failed_grip_prevents_lift(picker):
    p, a = picker
    advance_to_grip(p)
    p.service(16, FLANGE, 2, 16, GripFeedback(2, 16, 1000, (0,) * 6))
    assert p.stage == "cancelled"
    assert len(a.commands) == 4


def test_preinsert_error_stops(picker):
    p, a = picker
    p.stage, p.deadline = "verify", 5
    p.service(1, FLANGE + [0.02, 0, 0, 0, 0, 0, 0], 1, 1, None)
    assert a.errors and "verification failed" in a.errors[0]


def test_repeated_start_ignored(picker):
    p, a = picker
    advance_to_grip(p)
    p.start(1, 100)
    assert p.stage == "grip"
