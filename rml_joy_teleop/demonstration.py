"""Measured observations and bounded, crash-recoverable demonstration storage.

This module has no ROS imports. Online consumers may omit gripper observations
with snapshot(now, last_image, False); teleop always uses the 19-state default.
"""
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid
import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .auto_pick import TCP_OFFSET, pose_array, tcp_position

MAX_AGE = .25
STATE_KEYS = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")
STATE_COMPONENTS = (
    "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz",
    "tcp_vx", "tcp_vy", "tcp_vz", "tcp_wx", "tcp_wy", "tcp_wz",
    "force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z",
    "gripper_joint_position",
)


def source_stamp(msg):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value else None


def validate_image(payload):
    width, height, encoding, step, data = payload
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if encoding not in channels:
        raise ValueError("Unsupported RGB image encoding: " + str(encoding))
    if width <= 0 or height <= 0 or step < width * channels[encoding] or len(data) != height * step:
        raise ValueError("Invalid image dimensions/stride/data length")


def rgb_image(payload):
    validate_image(payload)
    width, height, encoding, step, data = payload
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}[encoding]
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    pixels = pixels[:, :width * channels].reshape(height, width, channels)
    if encoding.startswith("bgr"):
        pixels = pixels[:, :, [2, 1, 0]]
    elif encoding == "mono8":
        pixels = np.repeat(pixels, 3, axis=2)
    else:
        pixels = pixels[:, :, :3]
    return cv2.resize(pixels, (128, 128), interpolation=cv2.INTER_AREA)


class ObservationBuffer:
    def __init__(self):
        self.cache = {}
        self.calibration = None
        self.velocity = None
        self.previous_pose = None
        self.previous_time = None

    def put(self, key, value, now, ros_ns, source_ns=None):
        previous = self.cache.get(key)
        self.cache[key] = dict(value=value, receipt_monotonic=now,
                               receipt_ros_ns=ros_ns, source_ns=source_ns,
                               sequence=1 if previous is None else previous["sequence"] + 1)

    def set_calibration(self, open_register, closed_register, closed_m):
        values = np.asarray([open_register, closed_register, closed_m], dtype=float)
        if not np.isfinite(values).all() or open_register == closed_register or closed_m <= 0:
            raise ValueError("Invalid measured gripper calibration")
        self.calibration = dict(open_register=float(open_register),
                                closed_register=float(closed_register), closed_m=float(closed_m))

    def pose(self, msg, now, ros_ns):
        try:
            flange = pose_array(msg)
            tcp = tcp_position(flange)
            rotation = Rotation.from_quat(flange[3:])
            self.velocity = None
            if self.previous_pose is not None:
                dt = now - self.previous_time
                if 1e-6 < dt <= MAX_AGE:
                    previous = self.previous_pose
                    linear = (tcp - tcp_position(previous)) / dt
                    angular = (rotation * Rotation.from_quat(previous[3:]).inv()).as_rotvec() / dt
                    self.velocity = np.r_[linear, angular].tolist()
            self.previous_pose, self.previous_time = flange, now
            self.put("pose", flange.tolist(), now, ros_ns, source_stamp(msg))
        except ValueError:
            self.velocity = None
            self.previous_pose = None
            self.cache.pop("pose", None)
            raise

    def snapshot(self, now, last_image=None, include_gripper=True):
        keys = ["pose", "wrench", "image", "coordinate"]
        if include_gripper:
            keys.append("gripper")
        missing = [key for key in keys if key not in self.cache or not
                   0 <= now - self.cache[key]["receipt_monotonic"] <= MAX_AGE]
        if missing:
            raise ValueError("Missing/stale recording feedback: " + ", ".join(missing))
        if self.velocity is None:
            raise ValueError("Missing valid measured velocity history")
        if include_gripper and self.calibration is None:
            raise ValueError("Missing measured gripper calibration")
        if last_image is not None and self.cache["image"]["sequence"] <= last_image:
            return None
        flange = self.cache["pose"]["value"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            angles = Rotation.from_quat(flange[3:]).as_euler("xyz")
        wrench = np.asarray(self.cache["wrench"]["value"], dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("Invalid raw wrench")
        coordinate = self.cache["coordinate"]["value"]
        if coordinate not in (0, 1, 2):
            raise ValueError("Invalid wrench coordinate")
        state = dict(tcp_pose=np.r_[tcp_position(flange), angles].tolist(),
                     tcp_vel=list(self.velocity), tcp_force=wrench[:3].tolist(),
                     tcp_torque=wrench[3:].tolist())
        result = dict(state=state, flange_pose_xyzw=list(flange),
                      wrench_coordinate=coordinate,
                      image_payload=self.cache["image"]["value"],
                      sample_monotonic=now, wall_time_ns=time.time_ns())
        if include_gripper:
            c = self.calibration
            raw = self.cache["gripper"]["value"]
            if not math.isfinite(raw) or not min(c["open_register"], c["closed_register"]) <= raw <= max(c["open_register"], c["closed_register"]):
                raise ValueError("Gripper feedback outside calibrated range")
            state["gripper_pose"] = [(raw - c["open_register"]) /
                                     (c["closed_register"] - c["open_register"]) * c["closed_m"]]
            result["raw_gripper_register"] = raw
        result["timing"] = {
            key: dict((k, v) for k, v in self.cache[key].items() if k != "value")
            for key in keys
        }
        for value in result["timing"].values():
            value["age"] = now - value["receipt_monotonic"]
        return result


class Scheduler:
    def __init__(self, rate=15.):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("record_rate_hz must be finite and positive")
        self.period = 1. / rate
        self.next = None

    def start(self, now):
        self.next = now + self.period

    def due(self, now):
        if self.next is None or now < self.next:
            return False
        # Skip missed slots, never emit a burst of invented observations.
        self.next += (math.floor((now - self.next) / self.period) + 1) * self.period
        return True


def json_write(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


class EpisodeWriter:
    def __init__(self, root, metadata, capacity=256):
        self.root = Path(root).expanduser()
        self.identifier = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex
        self.path = self.root / "staging" / self.identifier
        self.queue = queue.Queue(maxsize=capacity)
        self.ready, self.done = threading.Event(), threading.Event()
        self.finish_event = threading.Event()
        self.error = None
        self.outcome = None
        self.final_path = None
        self.metadata = copy.deepcopy(metadata)
        self.thread = threading.Thread(target=self._run, name="teleop-writer", daemon=True)
        self.thread.start()

    def submit(self, kind, value):
        if self.error or self.done.is_set() or self.finish_event.is_set():
            raise RuntimeError(self.error or "Recorder is closing")
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full as error:
            raise RuntimeError("Recorder queue overflow") from error

    def finish(self, outcome):
        if not self.finish_event.is_set():
            self.outcome = copy.deepcopy(outcome)
            self.finish_event.set()

    def _run(self):
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path / "images").mkdir()
            json_write(self.path / "metadata.json", self.metadata)
            with open(self.path / "frames.jsonl", "w") as frames, open(self.path / "events.jsonl", "w") as events:
                self.ready.set()
                while not self.finish_event.is_set() or not self.queue.empty():
                    try:
                        kind, value = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if kind == "frame":
                        value = dict(value)
                        payload = value.pop("image_payload")
                        name = f"images/{value['frame_index']:06d}.png"
                        rgb = rgb_image(payload)
                        if not cv2.imwrite(str(self.path / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                            raise OSError("PNG write failed")
                        value["images"] = {"wrist_1": name}
                        stream = frames
                    else:
                        stream = events
                    stream.write(json.dumps(value, allow_nan=False) + "\n")
                    stream.flush()
                for stream in (frames, events):
                    os.fsync(stream.fileno())
            outcome = self.outcome
            json_write(self.path / "outcome.json", outcome)
            status = outcome["status"]
            if status == "discard":
                cancel = self.root / "cancelled"
                cancel.mkdir(parents=True, exist_ok=True)
                json_write(cancel / (self.identifier + ".json"), outcome)
                shutil.rmtree(self.path)
                self.final_path = cancel / (self.identifier + ".json")
            else:
                destination = self.root / status
                destination.mkdir(parents=True, exist_ok=True)
                self.final_path = destination / self.identifier
                os.replace(self.path, self.final_path)
        except Exception as error:
            self.error = f"Recorder failure: {error}"
            # A partial staging directory is deliberately retained for recovery.
        finally:
            self.done.set()


class Recorder:
    def __init__(self, root, config, sensors):
        self.sensors = sensors
        self.scheduler = Scheduler(config["record_rate_hz"])
        metadata = dict(schema="rm63-teleop-v1", rate_hz=config["record_rate_hz"],
                        state_keys=STATE_KEYS, state_components=STATE_COMPONENTS,
                        tcp_offset_m=TCP_OFFSET.tolist(), euler_convention="extrinsic_xyz_radians",
                        velocity_frame="driver_pose_reference", wrench_frame="reported_coordinate",
                        action_units="commanded_xy_displacement_metres", configuration=config,
                        gripper_calibration=sensors.calibration,
                        pose_topic="/rm_driver/udp_arm_position", pose_reference="driver_work_frame",
                        pose_assumption="flange feedback with unchanged identity work frame")
        self.writer = EpisodeWriter(root, metadata)
        self.active = False
        self.started = False
        self.segment = -1
        self.index = 0
        self.last_image = None
        self.last_sample = None
        self.xy = np.zeros(2)
        self.compatible = True
        self.last_target = None
        self.terminal = False

    def event(self, kind, now, **values):
        self.writer.submit("event", dict(type=kind, monotonic=now, **values))

    def seed(self, target):
        self.last_target = np.asarray(target, dtype=float).copy()

    def command(self, target, mode, now, desired_force=0., suppressed=None, published=None):
        target = np.asarray(target, dtype=float)
        if self.active and self.last_target is not None:
            self.xy += target[:2] - self.last_target[:2]
            rotation_changed = np.linalg.norm(
                (Rotation.from_quat(target[3:]) * Rotation.from_quat(self.last_target[3:]).inv()).as_rotvec()
            ) > 1e-10
            self.compatible &= not rotation_changed and (mode == "hybrid" or abs(target[2] - self.last_target[2]) < 1e-10)
        self.seed(target)
        self.event("command", now, target_flange_xyzw=target.tolist(), mode=mode,
                   desired_force_z=desired_force, suppressed=suppressed, published=published)

    def capture(self, now, initial=False, terminal=False):
        sample = self.sensors.snapshot(now, self.last_image)
        if sample is None:
            return False
        sample.update(frame_index=self.index, segment_index=self.segment,
                      initial=initial, terminal=terminal,
                      dt=None if initial else now - self.last_sample,
                      action_xy_m=None if initial else self.xy.tolist(),
                      xy_compatible=bool(self.compatible))
        self.writer.submit("frame", sample)
        self.last_image = sample["timing"]["image"]["sequence"]
        self.last_sample = now
        self.index += 1
        self.xy[:] = 0
        self.compatible = True
        self.terminal |= terminal
        return True

    def resume(self, now):
        if not self.writer.ready.is_set() or self.writer.error:
            return False
        if self.sensors.snapshot(now, self.last_image) is None:
            return False
        self.segment += 1
        self.xy[:] = 0
        self.compatible = True
        self.capture(now, initial=True)
        self.active = self.started = True
        self.scheduler.start(now)
        self.event("segment_start", now, segment_index=self.segment)
        return True

    def pause(self, now):
        if self.active:
            self.event("segment_end", now, segment_index=self.segment, truncation=True,
                       unobserved_action_xy_m=self.xy.tolist())
        self.active = False
        self.xy[:] = 0
        self.compatible = True

    def tick(self, now):
        if self.writer.error:
            raise RuntimeError(self.writer.error)
        # Paused episodes still require fresh feedback; no transitions span a pause.
        self.sensors.snapshot(now)
        if self.active and self.scheduler.due(now):
            self.capture(now)


def recover_staging(root):
    """Explicit offline recovery; call only after stopping all writers."""
    root = Path(root).expanduser()
    recovered = []
    for path in sorted((root / "staging").glob("*")):
        if not path.is_dir():
            continue
        destination = root / "incomplete" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        json_write(path / "outcome.json", dict(status="incomplete", requested=None,
                   stop_confirmed=False, terminal_observation=False, reason="recovered_staging"))
        os.replace(path, destination)
        recovered.append(str(destination))
    return recovered
