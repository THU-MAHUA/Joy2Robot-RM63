"""Nonblocking fixed-point picker. Its adapter owns ROS and outstanding results."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation

TCP_OFFSET = np.array([0., 0., .15875])
PREINSERT = np.array([.430, -.010, .03125])
LOWER = np.array([-.8, -.8, .02])
UPPER = np.array([.8, .8, 1.])


def pose_array(pose):
    values = np.array([pose.position.x, pose.position.y, pose.position.z,
                       pose.orientation.x, pose.orientation.y,
                       pose.orientation.z, pose.orientation.w], dtype=float)
    if not np.isfinite(values).all() or np.linalg.norm(values[3:]) < 1e-9:
        raise ValueError("Invalid measured flange pose")
    return values


def tcp_position(flange):
    return np.asarray(flange[:3]) + Rotation.from_quat(flange[3:]).apply(TCP_OFFSET)


def check_workspace(position):
    p = np.asarray(position)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < LOWER) or np.any(p > UPPER):
        raise ValueError("Flange target outside workspace: " + str(p))


def waypoints(flange, grab_mm=(281., -7.711, 158.), height=.03):
    flange = np.asarray(flange, dtype=float)
    if flange.shape != (7,) or not np.isfinite(flange).all() or height <= 0:
        raise ValueError("Invalid pickup geometry")
    rotation = Rotation.from_quat(flange[3:])
    offset = rotation.apply(TCP_OFFSET)
    grab = np.asarray(grab_mm, dtype=float) / 1000
    if grab.shape != (3,) or not np.isfinite(grab).all():
        raise ValueError("Invalid taught grab flange XYZ")
    tcp = grab + offset
    if tcp[2] > .050:
        raise ValueError("Vertical clearance would descend from grab TCP")
    points = dict(
        approach=grab + [0, 0, height], grab=grab,
        lift=np.array([tcp[0], tcp[1], .050]) - offset,
        transfer=np.array([.430, -.010, .050]) - offset,
        preinsert=PREINSERT - offset,
    )
    check_workspace(flange[:3])
    for p in points.values():
        check_workspace(p)
    return points, rotation.as_quat()


@dataclass
class GripFeedback:
    sequence: int
    received: float
    position: int
    tactile: tuple


class Picker:
    """Adapter: motion(kind, xyz, q), gripper(position, duration), fail(reason)."""
    def __init__(self, adapter, grab_mm=(281., -7.711, 158.), height=.03):
        self.adapter = adapter
        self.grab_mm, self.height = grab_mm, height
        self.stage = "idle"
        self.deadline = 0.
        self.pose_sequence = 0
        self.baseline = None
        self.grip_sequence = 0
        self.tactile_count = self.position_count = 0
        self.completed = None

    @property
    def active(self):
        return self.stage not in ("idle", "done", "cancelled")

    def start(self, now, pose_sequence):
        if self.active:
            return
        self.completed = None
        self.pose_sequence = pose_sequence
        self.stage, self.deadline = "fresh_pose", now + 5

    def cancel(self):
        self.stage = "cancelled"

    def _gripper(self, stage, position, duration, now):
        self.stage, self.deadline = stage, now + 90
        self.adapter.gripper(position, duration)

    def _motion(self, stage, now):
        self.stage, self.deadline = stage, now + 90
        self.adapter.motion("movej" if stage == "approach" else "movel",
                            self.points[stage], self.orientation)

    def result(self, success, now):
        if not self.active:
            return
        if not success:
            self.adapter.fail("Pickup command failed in " + self.stage)
            return
        if self.stage == "open":
            self._motion("approach", now)
        elif self.stage == "approach":
            self._motion("grab", now)
        elif self.stage == "grab":
            self.stage, self.deadline = "baseline", now + 3
        elif self.stage == "close":
            self.stage, self.deadline = "grip", now + 15
            self.tactile_count = self.position_count = 0
        elif self.stage == "lift":
            self._motion("transfer", now)
        elif self.stage == "transfer":
            self._motion("preinsert", now)
        elif self.stage == "preinsert":
            self.stage, self.deadline = "verify", now + 5
            self.pose_sequence = self.adapter.pose_sequence

    def tick(self, now, flange, pose_sequence, pose_time, grip):
        if not self.active:
            return
        if now > self.deadline:
            self.adapter.fail("Pickup timeout: " + self.stage)
            return
        if self.stage == "fresh_pose":
            if flange is None or pose_sequence <= self.pose_sequence:
                return
            if now - pose_time > .25:
                return
            self.points, self.orientation = waypoints(flange, self.grab_mm, self.height)
            self._gripper("open", .1, 1., now)
        elif self.stage == "baseline":
            if grip is not None and now - grip.received <= .25:
                self.baseline = grip
                self.grip_sequence = grip.sequence
                self._gripper("close", .014, 1.5, now)
        elif self.stage == "grip":
            if grip is None or now - grip.received > .25:
                return
            if grip.sequence <= self.grip_sequence:
                return
            self.grip_sequence = grip.sequence
            delta = np.linalg.norm(np.subtract(grip.tactile, self.baseline.tactile))
            # run_task.py's .014 joint target with command mapping 1000 -> 0.
            self.tactile_count = self.tactile_count + 1 if delta >= 20 else 0
            self.position_count = self.position_count + 1 if abs(grip.position - 860) <= 20 else 0
            if max(self.tactile_count, self.position_count) >= 2:
                self.stage, self.deadline = "settle", now + 1.5
        elif self.stage == "settle":
            pass
        elif self.stage == "verify" and pose_sequence > self.pose_sequence:
            if now - pose_time > .25:
                return
            error = float(np.linalg.norm(tcp_position(flange) - PREINSERT))
            if not math.isfinite(error) or error > .003:
                self.adapter.fail(f"Pre-insert TCP verification failed ({error * 1000:.3f} mm error)")
            else:
                self.stage, self.completed = "done", np.array(flange, copy=True)

    def service(self, now, *feedback):
        if self.stage == "settle" and now >= self.deadline:
            self._motion("lift", now)
        else:
            self.tick(now, *feedback)
