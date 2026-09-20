"""ROS adapter for pickup, hybrid ownership, stop acknowledgments and recording."""
import math
from time import monotonic

import numpy as np
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import ParameterEvent, ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rm_ros_interfaces.msg import Forcepositionmove, Movejp, Movel, Sixforce
from ros2_hkv_gripper.msg import GripperRegisters
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, UInt16
from trajectory_msgs.msg import JointTrajectoryPoint

from .auto_pick import Picker, GripFeedback, check_workspace, pose_array
from .demonstration import ObservationBuffer, Recorder, source_stamp, validate_image
from .hybrid_force import HybridForce


def fill_pose(msg, target):
    msg.position.x, msg.position.y, msg.position.z = map(float, target[:3])
    (msg.orientation.x, msg.orientation.y, msg.orientation.z,
     msg.orientation.w) = map(float, target[3:])


class TeleopWorkflow:
    def __init__(self, node):
        self.node = node
        defaults = dict(
            auto_pick_on_start=True, record_demonstrations=True, record_rate_hz=15.,
            record_camera_topic="/cameras/d435i/color/image_raw",
            record_dataset_root="~/vla_datasets/rm63_teleop",
            record_gripper_bridge_node="/realman_gripper_bridge",
            grab_ee_base_mm=[281., -7.711, 158.], pregrab_height=.03,
            force_topic="/rm_driver/udp_six_zero_force", joy_timeout_s=.25,
        )
        self.config = {}
        for key, default in defaults.items():
            node.declare_parameter(key, default)
            self.config[key] = node.get_parameter(key).value
        if self.config["record_demonstrations"] and not self.config["auto_pick_on_start"]:
            raise ValueError("record_demonstrations requires auto_pick_on_start")
        for key in ("record_rate_hz", "joy_timeout_s", "pregrab_height"):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(key + " must be finite and positive")
        grab = np.asarray(self.config["grab_ee_base_mm"])
        if grab.shape != (3,) or not np.isfinite(grab).all():
            raise ValueError("grab_ee_base_mm requires three finite values")
        self.config.update(publish_rate_hz=float(node.get_parameter("publish_rate_hz").value),
                           step_per_tick_m=node.step, rot_per_tick_rad=node.rot_step,
                           force_target=7., force_z_sign=-1., force_feedback_gain=.5,
                           sliding_force_limit=10., force_hard_limit=15.,
                           hybrid_stream_hz=50., hybrid_z_speed_limit=.002)
        self.mode = "idle"
        self.hybrid = "off"
        self.pending = set()
        self.failed_ack = False
        self.restart_fault = None
        self.deadline = 0.
        self.hybrid_deadline = 0.
        self.measured = None
        self.pose_time = None
        self.pose_sequence = 0
        self.joy_time = None
        self.grip = None
        self.rb = False
        self.previous_start = False
        self.neutral_seen = False
        self.handoff_gate = False
        self.suppress_gripper = False
        self.force = HybridForce()
        self.sensors = ObservationBuffer()
        self.picker = Picker(self, grab, self.config["pregrab_height"])
        self.recorder = None
        self.outcome = None
        self.terminal_deadline = None
        self.stop_resume = False
        self.goal = None
        self.last_control_tick = None
        self.target_updated = None
        self.next_hybrid = 0.
        self.last_warning = -math.inf
        self.calibration_future = None
        self.next_calibration = 0.
        self.calibration_deadline = 0.
        self.calibration_keys = ("hkv_feedback_open_register",
                                 "hkv_feedback_closed_register", "gripper_closed_position")
        self.bridge_node = self.config["record_gripper_bridge_node"].rstrip("/")
        self.stop_pub = node.create_publisher(Empty, "/rm_driver/move_stop_cmd", 10)
        self.start_pub = node.create_publisher(Empty, "/rm_driver/start_force_position_move_cmd", 10)
        self.hybrid_stop_pub = node.create_publisher(Empty, "/rm_driver/stop_force_position_move_cmd", 10)
        self.hybrid_pub = node.create_publisher(Forcepositionmove, "/rm_driver/force_position_move_cmd", 10)
        self.motion_pubs = {
            "movej": node.create_publisher(Movejp, "/rm_driver/movej_p_cmd", 10),
            "movel": node.create_publisher(Movel, "/rm_driver/movel_cmd", 10),
        }
        self.gripper_client = ActionClient(node, FollowJointTrajectory,
                                           "/gripper_controller/follow_joint_trajectory")
        self.parameter_client = None
        if self.config["record_demonstrations"]:
            self.parameter_client = node.create_client(GetParameters, self.bridge_node + "/get_parameters")
            node.create_subscription(ParameterEvent, "/parameter_events", self.parameter_cb, 10)
            node.create_subscription(Image, self.config["record_camera_topic"], self.image_cb,
                                     qos_profile_sensor_data)
            node.create_subscription(UInt16, "/rm_driver/udp_arm_coordinate",
                                     self.coordinate_cb, qos_profile_sensor_data)
        node.create_subscription(Sixforce, self.config["force_topic"], self.force_cb,
                                 qos_profile_sensor_data)
        node.create_subscription(GripperRegisters, "/gripper_registers", self.grip_cb,
                                 qos_profile_sensor_data)
        for key, topic in (
            ("movej", "/rm_driver/movej_p_result"), ("movel", "/rm_driver/movel_result"),
            ("arm_stop", "/rm_driver/move_stop_result"),
            ("hybrid_start", "/rm_driver/start_force_position_move_result"),
            ("hybrid_stop", "/rm_driver/stop_force_position_move_result"),
        ):
            node.create_subscription(Bool, topic,
                                     lambda msg, name=key: self.result(name, bool(msg.data)), 10)
        self.hybrid_timer = node.create_timer(.02, self.stream_hybrid)
        node.get_logger().info(
            "Hybrid teleop: contact >=2 N for 3 filtered samples; target=7 N, "
            "command cap=10 N, force stop=15 N. START picks to preinsert; "
                f"neutral controls then RB starts {self.config['record_rate_hz']:g} Hz recording. A=success X=failure B=discard "
            "during recording. RB release pauses and stops force motion."
            if self.config["record_demonstrations"] else
            "Hybrid teleop ready; START picks/arms, RB enables manual motion, B stops."
        )
        if self.config["auto_pick_on_start"]:
            node.get_logger().warning(
                "Pickup copies gripper actions 0.1 then 0.014 from run_task.py. "
                "These conflict with the bridge open/closed convention: verify direction "
                "under supervision before pickup. Software stop does not prove standstill."
            )

    def ros_ns(self):
        return self.node.get_clock().now().nanoseconds

    def warning(self, text):
        now = monotonic()
        if now - self.last_warning > 2:
            self.node.get_logger().warning(text)
            self.last_warning = now

    def target(self):
        n = self.node
        return np.array([n.tx, n.ty, n.tz, n.qx, n.qy, n.qz, n.qw])

    def seed(self, preserve_contact_reference=False):
        if self.measured is None or monotonic() - self.pose_time > .25:
            raise ValueError("Missing/stale measured pose")
        n = self.node
        target = self.measured.copy()
        if preserve_contact_reference:
            target[2:] = self.target()[2:]
        n.tx, n.ty, n.tz, n.qx, n.qy, n.qz, n.qw = map(float, target)
        n.have_seed = True
        self.target_updated = monotonic()
        if self.recorder:
            self.recorder.seed(self.target())

    def pose(self, msg):
        now = monotonic()
        try:
            self.sensors.pose(msg, now, self.ros_ns())
            self.measured = pose_array(msg)
            self.pose_time = now
            self.pose_sequence += 1
        except ValueError as error:
            self.measured = None
            if self.mode not in ("idle", "fault"):
                self.fail(str(error))

    def force_cb(self, msg):
        now = monotonic()
        try:
            wrench = [float(getattr(msg, "force_" + key)) for key in ("fx", "fy", "fz", "mx", "my", "mz")]
            if not np.isfinite(wrench).all():
                raise ValueError("Nonfinite wrench")
            self.sensors.put("wrench", wrench, now, self.ros_ns(), source_stamp(msg))
            self.force.update(wrench[2], now)
            if self.force.fault and self.mode not in ("idle", "fault", "stopping"):
                self.fail("Force hard stop (15 N filtered compression)")
        except ValueError as error:
            self.force.fault = True
            if self.mode not in ("idle", "fault"):
                self.fail(str(error))

    def grip_cb(self, msg):
        now = monotonic()
        sequence = 1 if self.grip is None else self.grip.sequence + 1
        self.grip = GripFeedback(sequence, now, int(msg.position), tuple(
            float(getattr(msg, key + "_value")) for key in
            ("x1", "y1", "z1", "x2", "y2", "z2")))
        self.sensors.put("gripper", int(msg.position), now, self.ros_ns(), source_stamp(msg))

    def image_cb(self, msg):
        try:
            payload = (msg.width, msg.height, msg.encoding, msg.step, bytes(msg.data))
            validate_image(payload)
            previous = self.sensors.cache.get("image")
            stamp = source_stamp(msg)
            if previous and stamp and previous["source_ns"] and stamp <= previous["source_ns"]:
                return
            layout = (msg.width, msg.height, msg.encoding, msg.header.frame_id)
            if self.recorder and layout != getattr(self, "image_layout", layout):
                raise ValueError("Recording camera layout/frame changed")
            self.image_layout = layout
            self.sensors.put("image", payload, monotonic(), self.ros_ns(), stamp)
        except ValueError as error:
            self.sensors.cache.pop("image", None)
            if self.recorder:
                self.fail(str(error))

    def coordinate_cb(self, msg):
        old = self.sensors.cache.get("coordinate")
        value = int(msg.data)
        if value not in (0, 1, 2) or (self.recorder and old and old["value"] != value):
            self.fail("Wrench coordinate changed or invalid")
            return
        self.sensors.put("coordinate", value, monotonic(), self.ros_ns(), source_stamp(msg))

    def parameter_cb(self, msg):
        if msg.node == self.bridge_node and any(p.name in self.calibration_keys for p in
                (*msg.new_parameters, *msg.changed_parameters, *msg.deleted_parameters)):
            self.sensors.calibration = None
            self.next_calibration = 0
            if self.recorder:
                self.fail("Gripper feedback calibration changed")

    def calibration(self, now):
        if self.parameter_client is None:
            return
        future = self.calibration_future
        if future:
            if future.done():
                self.calibration_future = None
                values = []
                for parameter in future.result().values:
                    if parameter.type == ParameterType.PARAMETER_INTEGER:
                        values.append(parameter.integer_value)
                    elif parameter.type == ParameterType.PARAMETER_DOUBLE:
                        values.append(parameter.double_value)
                    else:
                        raise ValueError("Missing deployed gripper calibration parameter")
                if len(values) != 3:
                    raise ValueError("Incomplete gripper calibration")
                old = self.sensors.calibration
                self.sensors.set_calibration(*values)
                if self.recorder and old != self.sensors.calibration:
                    raise ValueError("Gripper feedback calibration changed")
            elif now > self.calibration_deadline:
                future.cancel()
                self.calibration_future = None
                self.sensors.calibration = None
                raise ValueError("Gripper calibration service timeout")
        if self.calibration_future is None and now >= self.next_calibration:
            self.next_calibration = now + 1
            if self.parameter_client.service_is_ready():
                self.calibration_future = self.parameter_client.call_async(
                    GetParameters.Request(names=list(self.calibration_keys)))
                self.calibration_deadline = now + 2

    def motion(self, kind, position, orientation):
        if self.mode != "pick" or self.pending:
            raise RuntimeError("Pickup command ownership violation")
        check_workspace(position)
        message = Movejp() if kind == "movej" else Movel()
        fill_pose(message.pose, np.r_[position, orientation])
        message.speed, message.trajectory_connect, message.block = 10, 0, True
        self.pending.add(kind)
        self.motion_pubs[kind].publish(message)

    def gripper(self, position, duration):
        if self.mode != "pick" or self.pending:
            raise RuntimeError("Gripper action ownership violation")
        if not self.gripper_client.server_is_ready():
            raise RuntimeError("Gripper trajectory action server not ready")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["gripper_joint"]
        point = JointTrajectoryPoint(positions=[float(position)])
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points = [point]
        self.pending.add("gripper")
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        try:
            goal = future.result()
            if not goal.accepted:
                self.result("gripper", False)
                return
            self.goal = goal
            goal.get_result_async().add_done_callback(self.gripper_result)
            if self.mode != "pick":
                self.cancel_gripper()
        except Exception as error:
            self.restart_fault = "Unresolved gripper goal: " + str(error)
            self.fail(self.restart_fault)

    def gripper_result(self, future):
        try:
            result = future.result()
            self.goal = None
            success = result.status == GoalStatus.STATUS_SUCCEEDED and result.result.error_code == 0
            self.result("gripper", success)
        except Exception as error:
            self.restart_fault = "Unresolved gripper result: " + str(error)
            self.fail(self.restart_fault)

    def cancel_gripper(self):
        if self.goal is not None and "gripper_cancel" not in self.pending:
            self.pending.add("gripper_cancel")
            self.goal.cancel_goal_async().add_done_callback(self.cancel_result)

    def cancel_result(self, future):
        try:
            response = future.result()
            # ERROR_GOAL_TERMINATED is resolved by the separately awaited result.
            if response.return_code not in (0, 3):
                self.restart_fault = "Gripper cancellation rejected"
            self.pending.discard("gripper_cancel")
        except Exception as error:
            self.restart_fault = "Unresolved gripper cancellation: " + str(error)

    def result(self, key, success):
        if key not in self.pending:
            return
        self.pending.remove(key)
        now = monotonic()
        if self.mode in ("stopping", "fault"):
            if key in ("arm_stop", "hybrid_stop") and not success:
                self.failed_ack = True
                self.restart_fault = key + " acknowledgment failed"
            if key == "hybrid_start" and success:
                # A start accepted after the first stop must be stopped again.
                self.pending.add("hybrid_stop")
                self.hybrid_stop_pub.publish(Empty())
            return
        if key == "hybrid_start":
            if not success:
                self.fail("Hybrid start rejected")
            else:
                self.hybrid = "active"
                self.target_updated = now
        elif key in ("movej", "movel", "gripper"):
            try:
                self.picker.result(success, now)
            except Exception as error:
                self.fail(str(error))

    def fail(self, reason):
        self.stop(reason, requested="incomplete")

    def latch_outcome(self, requested, reason, now):
        if self.outcome is not None:
            return
        paused = self.recorder is not None and not self.recorder.active
        self.outcome = dict(requested=requested, reason=reason, requested_monotonic=now,
                            outcome_while_paused=paused, fault=None,
                            terminal_observation=False, stop_confirmed=False)
        self.stop_resume = False
        self.terminal_deadline = None
        if self.recorder:
            if paused and requested in ("success", "failure"):
                try:
                    # Validate feedback without creating a transition across the pause.
                    self.sensors.snapshot(now)
                except ValueError as error:
                    self.outcome["fault"] = str(error)
            if self.recorder.active and requested in ("success", "failure"):
                self.terminal_deadline = now + .25
                try:
                    self.recorder.capture(now, terminal=True)
                except (ValueError, RuntimeError) as error:
                    self.outcome["fault"] = str(error)
            self.recorder.active = False
        self.suppress_gripper = True

    def stop(self, reason, requested=None, resume=False):
        if self.mode in ("stopping", "fault"):
            if self.mode == "stopping" and requested is not None and self.outcome is None:
                self.latch_outcome(requested, reason, monotonic())
            if requested == "incomplete":
                self.stop_resume = False
                if self.outcome:
                    self.outcome["fault"] = reason
            return
        now = monotonic()
        self.node.armed = False
        self.node._zero_velocities()
        self.stop_resume = resume
        self.mode = "stopping"
        self.hybrid = "off"
        self.target_updated = None
        self.deadline = now + 2
        self.picker.cancel()
        self.failed_ack = False
        if requested is not None:
            self.latch_outcome(requested, reason, now)
        self.pending.update(("arm_stop", "hybrid_stop"))
        self.stop_pub.publish(Empty())
        self.hybrid_stop_pub.publish(Empty())
        self.cancel_gripper()
        if self.recorder:
            try:
                self.recorder.event("stop_requested", now, requested=requested, reason=reason,
                                    topics=["/rm_driver/move_stop_cmd",
                                            "/rm_driver/stop_force_position_move_cmd"])
            except RuntimeError as error:
                if self.outcome:
                    self.outcome["fault"] = str(error)
                self.stop_resume = False
                self.restart_fault = str(error)
        self.node.get_logger().info("Stop requested: " + reason)

    def prepare_recorder(self, now):
        if not self.config["record_demonstrations"] or self.recorder is not None:
            return
        self.sensors.snapshot(now)
        config = dict(self.config, camera_layout=getattr(self, "image_layout", None))
        self.recorder = Recorder(self.config["record_dataset_root"], config, self.sensors)

    def finish_stop(self, now):
        if (self.recorder and self.outcome and self.terminal_deadline is not None
                and not self.recorder.terminal and not self.outcome["fault"]):
            try:
                if now <= self.terminal_deadline:
                    self.recorder.capture(now, terminal=True)
            except (ValueError, RuntimeError) as error:
                self.outcome["fault"] = str(error)
            if not self.recorder.terminal and not self.outcome["fault"] and now < self.terminal_deadline:
                return
        unresolved = bool(self.pending)
        if unresolved and now < self.deadline:
            return
        if unresolved:
            self.restart_fault = "Missing acknowledgments/results: " + ", ".join(sorted(self.pending))
        bad = unresolved or self.failed_ack or self.restart_fault or self.force.fault
        if self.stop_resume and not bad:
            self.mode = "manual"
            self.node.armed = True
            self.seed(preserve_contact_reference=self.force.contact)
            self.handoff_gate = True
            self.neutral_seen = False
            self.stop_resume = False
            return
        if self.outcome and self.recorder:
            writer = self.recorder.writer
            if not writer.finish_event.is_set():
                terminal = self.recorder.terminal
                requested = self.outcome["requested"]
                paused = self.outcome["outcome_while_paused"]
                valid = not bad and not self.outcome["fault"] and not writer.error
                # Paused outcomes are episode labels, never artificial terminal transitions.
                valid &= terminal or paused
                status = requested if requested in ("success", "failure") and valid else "incomplete"
                if requested == "discard":
                    status = "discard"
                self.outcome.update(status=status, stop_confirmed=not bad,
                                    terminal_observation=terminal, frames=self.recorder.index,
                                    segments=self.recorder.segment + 1,
                                    fault=self.outcome["fault"] or self.restart_fault or writer.error or
                                    ("Missing terminal observation" if not terminal and not paused and requested != "discard" else None))
                writer.finish(self.outcome)
            if not writer.done.is_set():
                return
            if writer.error:
                self.restart_fault = writer.error
                bad = True
            self.node.get_logger().info(f"Episode {self.outcome['status']}: {writer.final_path or writer.path}")
            self.recorder = None
        self.mode = "fault" if bad else "idle"
        self.node.armed = False
        self.outcome = None
        self.stop_resume = False
        self.handoff_gate = False
        if bad:
            self.warning("Restart required: " + str(self.restart_fault or "latched force fault"))

    def ready_feedback(self, now):
        if self.measured is None or self.pose_time is None or now - self.pose_time > .25:
            raise ValueError("Missing/stale pose feedback")
        if self.force.received is None or now - self.force.received > .25:
            raise ValueError("Missing/stale force feedback")
        if self.joy_time is None or now - self.joy_time > self.config["joy_timeout_s"]:
            raise ValueError("Missing/stale joystick feedback")
        if self.force.fault:
            raise ValueError("Latched force fault; restart required")

    def neutral(self, msg):
        axes = msg.axes
        return (all(abs(axes[i]) < self.node.dz for i in (0, 1, 4, 6, 7)) and
                all(abs(1 - axes[i]) / 2 < self.node.dz for i in (2, 5)) and
                not any(msg.buttons[i] for i in (0, 1, 2, 3, 8)))

    def joy(self, msg):
        """Return true to consume the input before legacy mappings."""
        now = monotonic()
        # Outcome buttons are independent of RB and of otherwise malformed axes.
        if len(msg.buttons) > 2:
            if self.recorder and self.recorder.started and self.outcome is None:
                requested = "discard" if msg.buttons[1] else "failure" if msg.buttons[2] else "success" if msg.buttons[0] else None
                if requested:
                    self.previous_start = bool(msg.buttons[7]) if len(msg.buttons) > 7 else False
                    self.stop("operator_" + requested, requested=requested)
                    try:
                        self.recorder.event("outcome_buttons", now, buttons=list(msg.buttons),
                                            selected=requested)
                    except RuntimeError as error:
                        self.outcome["fault"] = str(error)
                    return True
            if msg.buttons[1]:
                self.previous_start = bool(msg.buttons[7]) if len(msg.buttons) > 7 else False
                if self.mode not in ("idle", "fault", "stopping"):
                    self.stop("operator_cancel", requested="discard" if self.recorder else None)
                return True
        if (len(msg.buttons) < 9 or len(msg.axes) < 8 or not np.isfinite(msg.axes).all()
                or any(abs(v) > 1.00001 for v in msg.axes)
                or any(v not in (0, 1) for v in msg.buttons)):
            if self.mode not in ("idle", "fault"):
                self.fail("Malformed joystick input")
            return True
        start = bool(msg.buttons[7])
        start_edge = start and not self.previous_start
        self.previous_start = start
        previous_rb = self.rb
        self.rb = bool(msg.buttons[5])
        self.joy_time = now
        outcome_buttons = any(msg.buttons[i] for i in (0, 1, 2))
        if not outcome_buttons:
            self.suppress_gripper = False
        if self.mode in ("stopping", "fault", "pick"):
            return True
        if self.recorder and self.recorder.started:
            try:
                self.recorder.event("joystick", now, axes=list(msg.axes), buttons=list(msg.buttons),
                                    suppressed=("Z/rotation/HOME/Y" if self.force.contact else "HOME/Y"))
            except RuntimeError as error:
                self.fail(str(error))
                return True
        if self.mode == "idle":
            if start_edge and not self.suppress_gripper:
                try:
                    self.ready_feedback(now)
                    self.seed()
                    if self.config["auto_pick_on_start"]:
                        self.mode = "pick"
                        self.picker.start(now, self.pose_sequence)
                    else:
                        self.mode, self.node.armed = "manual", True
                except ValueError as error:
                    self.warning(str(error))
                return True
            return self.suppress_gripper
        if self.mode == "handoff" or self.handoff_gate:
            if not self.rb and self.neutral(msg):
                self.neutral_seen = True
            if not (self.rb and not previous_rb and self.neutral_seen and self.neutral(msg)):
                return True
            try:
                self.ready_feedback(now)
                if self.config["record_demonstrations"]:
                    self.sensors.snapshot(now)
                    if self.recorder is None:
                        self.prepare_recorder(now)
                        self.warning("Preparing recorder; release RB then press again when ready")
                        return True
                    if not self.recorder.resume(now):
                        self.warning("Recorder awaiting readiness or a new image; release and press RB")
                        return True
                self.seed(preserve_contact_reference=self.force.contact)
                self.node._zero_velocities()
                self.node.deadman = True
                self.mode, self.node.armed = "manual", True
                self.handoff_gate = self.neutral_seen = False
            except (ValueError, RuntimeError) as error:
                self.warning(str(error))
                return True
        if self.mode != "manual":
            return True
        if previous_rb and not self.rb:
            if self.recorder:
                self.recorder.pause(now)
            if self.hybrid != "off" or self.recorder:
                self.stop("RB released", resume=True)
                return True
        if self.suppress_gripper:
            return True
        return False

    def tick(self):
        """Return true when position streaming must be suppressed."""
        now = monotonic()
        try:
            try:
                self.calibration(now)
            except Exception as error:
                if self.recorder:
                    raise
                self.warning(str(error))
            if self.mode == "stopping":
                self.finish_stop(now)
                return True
            if self.mode in ("idle", "fault"):
                return True
            self.ready_feedback(now)
            if self.recorder:
                if self.recorder.writer.error:
                    raise RuntimeError(self.recorder.writer.error)
                if self.recorder.started:
                    self.recorder.tick(now)
            if self.mode == "pick":
                self.picker.service(now, self.measured, self.pose_sequence, self.pose_time, self.grip)
                if self.picker.stage == "done":
                    self.seed()
                    self.node._zero_velocities()
                    self.force.reset_contact()
                    self.mode = "handoff"
                    self.handoff_gate = True
                    self.neutral_seen = False
                    self.node.get_logger().info("Verified preinsert. Release RB and neutralize controls, then press RB.")
                return True
            if self.mode == "handoff" or self.handoff_gate:
                try:
                    self.prepare_recorder(now)
                except ValueError as error:
                    self.warning(str(error))
                return True
            if self.force.contact:
                if not self.node.deadman:
                    return True
                if self.hybrid == "off":
                    self.hybrid = "starting"
                    self.pending.add("hybrid_start")
                    self.hybrid_deadline = now + 2
                    self.start_pub.publish(Empty())
                    self.last_control_tick = now
                elif self.hybrid == "starting":
                    if now > self.hybrid_deadline:
                        raise ValueError("Hybrid start timeout")
                else:
                    # Scale per-tick displacement to time; never integrate through a delayed tick.
                    dt = now - self.last_control_tick
                    if dt > .25:
                        raise ValueError("Hybrid target update stalled")
                    scale = min(dt, 1. / self.config["publish_rate_hz"]) * self.config["publish_rate_hz"]
                    self.node.tx += self.node.step * self.node.vx * scale
                    self.node.ty += self.node.step * self.node.vy * scale
                    check_workspace(self.target()[:3])
                    self.target_updated = now
                self.last_control_tick = now
                return True
            check_workspace(self.target()[:3])
            return False
        except Exception as error:
            self.fail(str(error))
            return True

    def stream_hybrid(self):
        if self.mode != "manual" or self.hybrid != "active":
            return
        now = monotonic()
        try:
            self.ready_feedback(now)
            if not self.node.deadman:
                return
            if self.target_updated is None or now - self.target_updated > .25:
                raise ValueError("Hybrid target became stale")
            target = self.target()
            check_workspace(target[:3])
            msg = Forcepositionmove()
            fill_pose(msg.pose, target)
            msg.flag, msg.sensor, msg.mode, msg.follow = 1, 1, 0, False
            msg.control_mode = [3, 3, 4, 0, 0, 0]
            msg.desired_force[2] = -self.force.commanded_compression()
            msg.limit_vel[2] = .002
            self.hybrid_pub.publish(msg)
            self.published("hybrid", float(msg.desired_force[2]), msg)
        except Exception as error:
            self.fail(str(error))

    def published(self, mode="position", desired_force=0., message=None):
        if self.recorder:
            try:
                self.recorder.command(
                    self.target(), mode, monotonic(), desired_force,
                    suppressed="Z/rotation" if mode == "hybrid" else None,
                    published=message_to_ordereddict(message) if message is not None else None)
            except RuntimeError as error:
                self.fail(str(error))

    def position_allowed(self):
        try:
            check_workspace(self.target()[:3])
            return True
        except ValueError as error:
            self.fail(str(error))
            return False

    @property
    def allow_gripper(self):
        return self.mode in ("idle", "manual") and not self.suppress_gripper and self.recorder is None

    @property
    def allow_home(self):
        return self.allow_gripper and not self.force.contact

    def shutdown(self):
        if self.mode not in ("idle", "fault"):
            self.stop("shutdown", requested="incomplete")
