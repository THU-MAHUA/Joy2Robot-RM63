"""ROS messages are real; nodes, publishers, timers and action clients are mocks."""
import copy
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Joy
from rclpy.node import Node

from rml_joy_teleop import teleop_workflow as workflow
from rml_joy_teleop.xbox_xyz_teleop import XboxXYZTeleop


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(copy.deepcopy(msg))


class Action:
    def __init__(self, *args):
        self.requests = []
        self.future = None

    def server_is_ready(self):
        return True

    def send_goal_async(self, request):
        self.requests.append(request)
        self.future = Future()
        return self.future


def pose(x=.430, y=-.010, z=.190, q=(1., 0., 0., 0.)):
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = float(x), float(y), float(z)
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
    return msg


def joy(*buttons, axes=None):
    msg = Joy()
    msg.axes = list(axes if axes is not None else (0., 0., 1., 0., 0., 1., 0., 0.))
    msg.buttons = [int(i in buttons) for i in range(11)]
    return msg


@pytest.fixture
def node_factory(monkeypatch, tmp_path):
    nodes = []
    clock = [10.]

    def initialize(self, name):
        self.params = dict(record_dataset_root=str(tmp_path),
                           record_demonstrations=False, auto_pick_on_start=False)
        self.params.update(overrides)
        self.mock_publishers, self.mock_timers, self.mock_subscriptions = {}, [], {}
        self.logs = []

    def declare(self, name, default):
        self.params.setdefault(name, default)

    def publisher(self, kind, topic, qos):
        p = Publisher()
        self.mock_publishers[topic] = p
        return p

    def subscription(self, kind, topic, callback, qos):
        self.mock_subscriptions[topic] = callback
        return callback

    def timer(self, period, callback):
        self.mock_timers.append((period, callback))
        return callback

    monkeypatch.setattr(Node, "__init__", initialize)
    monkeypatch.setattr(Node, "declare_parameter", declare)
    monkeypatch.setattr(Node, "get_parameter", lambda self, name: SimpleNamespace(value=self.params[name]))
    monkeypatch.setattr(Node, "create_publisher", publisher)
    monkeypatch.setattr(Node, "create_subscription", subscription)
    monkeypatch.setattr(Node, "create_timer", timer)
    monkeypatch.setattr(Node, "create_client", lambda *a: SimpleNamespace(service_is_ready=lambda: False))
    monkeypatch.setattr(Node, "get_clock", lambda self: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=int(clock[0] * 1e9))))
    monkeypatch.setattr(Node, "get_logger", lambda self: SimpleNamespace(
        info=self.logs.append, warning=self.logs.append, warn=self.logs.append, error=self.logs.append))
    monkeypatch.setattr(workflow, "ActionClient", Action)
    monkeypatch.setattr(workflow, "monotonic", lambda: clock[0])
    overrides = {}

    def make(**parameters):
        overrides.clear()
        overrides.update(parameters)
        node = XboxXYZTeleop()
        nodes.append(node)
        node.pose_cb(pose())
        clock[0] += .01
        node.pose_cb(pose())
        node.workflow.force.update(0., clock[0])
        node.joy_cb(joy())
        return node, clock

    yield make
    for node in nodes:
        recorder = node.workflow.recorder
        if recorder:
            recorder.writer.finish(dict(status="incomplete", reason="test_cleanup"))
            recorder.writer.thread.join(5)
