#!/usr/bin/env python3
"""
Xbox joystick 6-DOF teleop for RML63 using pose CANFD passthrough.

Design notes
------------
- Arm motion requires the RB deadman button.
- Releasing all inputs holds the arm in place.
- START picks to verified preinsert by default; neutral then RB hands off.
- Recording starts at that handoff: A success, X failure, B discard.
- Contact switches to acknowledged 50 Hz XY/force-Z control.
- The driver requires *continuous* streaming on /rm_driver/movep_canfd_cmd
  so we publish at a fixed 100 Hz timer regardless of stick activity.
- Pose is seeded from /rm_driver/udp_arm_position before arming so the
  first passthrough point is coincident with the real arm pose.
- Gripper commands default to the robot-end extension bridge used by
  rm_63_gripper_bringup.launch.py. Direct USB control remains selectable.

Controller mapping
------------------
  Hold RB (R1)               -> Deadman (required for arm motion)
  Left  stick  up/down       -> X translation
  Left  stick  left/right    -> Y translation
  Right stick  up/down       -> Z translation
  D-pad left/right           -> Yaw
  D-pad up/down              -> Pitch
  RT trigger                 -> Roll +
  LT trigger                 -> Roll -
  Y button                   -> Gripper open  (0.0 m)
  X button                   -> Gripper half-open (0.044 m)
  A button                   -> Gripper close (0.1 m)
  START                      -> Arm
  B                          -> Disarm
"""

import math
from time import monotonic
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from rm_ros_interfaces.msg import Cartepos, Gripperset
from .teleop_workflow import TeleopWorkflow


# ---------------------------------------------------------------------------
# Xbox button / axis indices (standard Linux joy mapping)
# ---------------------------------------------------------------------------
BTN_A        = 0   # Gripper close
BTN_X        = 2   # Gripper half-open
BTN_Y        = 3   # Gripper fully open
BTN_B        = 1
BTN_RB       = 5
BTN_HOME     = 8   # Xbox / Home button
BTN_START    = 7
AXIS_DPAD_X  = 6   # d-pad horizontal: +1=left (yaw+), -1=right (yaw-)
AXIS_DPAD_Y  = 7   # d-pad vertical:   +1=up  (pitch+), -1=down (pitch-)

AXIS_LEFT_X  = 0
AXIS_LEFT_Y  = 1
AXIS_RIGHT_Y = 4
AXIS_LT      = 2   # 1=idle, -1=full press
AXIS_RT      = 5   # 1=idle, -1=full press


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def axis_angle_to_quat(ax, ay, az, angle):
    norm = math.sqrt(ax*ax + ay*ay + az*az)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(angle / 2.0) / norm
    return (ax*s, ay*s, az*s, math.cos(angle / 2.0))


def quat_normalise(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)

def euler_xyz_to_quat(rx, ry, rz):
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cz, sz = math.cos(rz/2), math.sin(rz/2)

    qx = sx*cy*cz - cx*sy*sz
    qy = cx*sy*cz + sx*cy*sz
    qz = cx*cy*sz - sx*sy*cz
    qw = cx*cy*cz + sx*sy*sz

    return quat_normalise((qx, qy, qz, qw))


def position_m_to_register(
    position_m,
    open_position_m,
    closed_position_m,
    open_register,
    closed_register,
):
    position_span = closed_position_m - open_position_m
    if abs(position_span) < 1e-9:
        return int(open_register)

    closed_fraction = (position_m - open_position_m) / position_span
    closed_fraction = max(0.0, min(1.0, closed_fraction))
    register = open_register + (closed_register - open_register) * closed_fraction
    return max(0, min(0xFFFF, int(round(register))))


class XboxXYZTeleop(Node):
    def __init__(self):
        super().__init__("xbox_xyz_teleop")

        # --- parameters ---
        self.declare_parameter("publish_rate_hz",  100.0)
        self.declare_parameter("step_per_tick_m",  0.0002)   # 2 cm/s at full stick @ 100 Hz
        self.declare_parameter("rot_per_tick_rad", 0.001)    # ~5.7 deg/s at full input @ 100 Hz
        self.declare_parameter("deadzone",         0.20)
        self.declare_parameter("follow_high",      False)
        self.declare_parameter("gripper_open_m",   0.0)      # Y: fully open
        self.declare_parameter("gripper_half_m",   0.044)    # X: half open
        self.declare_parameter("gripper_close_m",  0.1)      # A: fully closed
        self.declare_parameter("gripper_backend", "robot_end")
        self.declare_parameter("gripper_bridge_topic", "/gripper/set_position_cmd")
        self.declare_parameter(
            "gripper_usb_topic", "/tg9801_gripper_controller/commands"
        )
        self.declare_parameter("gripper_bridge_open_register", 1000)
        self.declare_parameter("gripper_bridge_closed_register", 0)
        self.declare_parameter("gripper_bridge_block", False)
        self.declare_parameter("gripper_bridge_timeout_ms", 0)

        rate             = float(self.get_parameter("publish_rate_hz").value)
        self.step        = float(self.get_parameter("step_per_tick_m").value)
        self.rot_step    = float(self.get_parameter("rot_per_tick_rad").value)
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(self.step) or not math.isfinite(self.rot_step):
            raise ValueError("Invalid position stream rate or step")
        self.dz          = float(self.get_parameter("deadzone").value)
        self.follow_high = bool(self.get_parameter("follow_high").value)
        self.gripper_open_m  = float(self.get_parameter("gripper_open_m").value)
        self.gripper_half_m  = float(self.get_parameter("gripper_half_m").value)
        self.gripper_close_m = float(self.get_parameter("gripper_close_m").value)
        self.gripper_backend = str(
            self.get_parameter("gripper_backend").value
        ).strip().lower()
        if self.gripper_backend in ("bridge", "extension", "end_extension"):
            self.gripper_backend = "robot_end"
        elif self.gripper_backend in ("direct_usb", "desktop_usb"):
            self.gripper_backend = "usb"
        if self.gripper_backend not in ("robot_end", "usb"):
            raise ValueError(
                "gripper_backend must be 'robot_end' or 'usb', got "
                f"'{self.gripper_backend}'"
            )
        self.gripper_bridge_topic = str(
            self.get_parameter("gripper_bridge_topic").value
        )
        self.gripper_usb_topic = str(
            self.get_parameter("gripper_usb_topic").value
        )
        self.gripper_bridge_open_register = int(
            self.get_parameter("gripper_bridge_open_register").value
        )
        self.gripper_bridge_closed_register = int(
            self.get_parameter("gripper_bridge_closed_register").value
        )
        self.gripper_bridge_block = bool(
            self.get_parameter("gripper_bridge_block").value
        )
        self.gripper_bridge_timeout_ms = max(
            0,
            min(
                0xFFFF,
                int(self.get_parameter("gripper_bridge_timeout_ms").value),
            ),
        )

        # --- state ---
        self.armed     = False
        self.have_seed = False

        self.tx = 0.0;  self.ty = 0.0;  self.tz = 0.0
        self.qx = 0.0;  self.qy = 0.0;  self.qz = 0.0;  self.qw = 1.0

        # --- HOME pose ---
        self.home_tx = 0.230196
        self.home_ty = 0.000032
        self.home_tz = 0.340558

        self.home_qx, self.home_qy, self.home_qz, self.home_qw = euler_xyz_to_quat(
            3.141, 0.003, 3.141
        )

        # velocity commands — zero means stopped
        self.vx      = 0.0;  self.vy      = 0.0;  self.vz    = 0.0
        self.v_roll  = 0.0;  self.v_pitch = 0.0;  self.v_yaw = 0.0
        self.deadman = False  # RB (R1) held

        # gripper edge-trigger tracking (publish once per button press)
        self._prev_x = 0
        self._prev_y = 0
        self._prev_a = 0

        # --- I/O ---
        self.pub = self.create_publisher(Cartepos, "/rm_driver/movep_canfd_cmd", 10)
        self.gripper_bridge_pub = None
        self.gripper_usb_pub = None
        if self.gripper_backend == "robot_end":
            self.gripper_bridge_pub = self.create_publisher(
                Gripperset, self.gripper_bridge_topic, 10
            )
            gripper_output = (
                f"robot-end extension via {self.gripper_bridge_topic} "
                f"(open={self.gripper_bridge_open_register}, "
                f"closed={self.gripper_bridge_closed_register})"
            )
        else:
            self.gripper_usb_pub = self.create_publisher(
                Float64MultiArray, self.gripper_usb_topic, 10
            )
            gripper_output = f"direct USB via {self.gripper_usb_topic}"
        self.create_subscription(Joy,  "/joy",                        self.joy_cb,  10)
        self.create_subscription(Pose, "/rm_driver/udp_arm_position", self.pose_cb, 10)
        self.workflow = TeleopWorkflow(self)
        self.timer = self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f"Waiting for /rm_driver/udp_arm_position to seed pose.\n"
            f"  START              = pick/preinsert (arm if auto_pick_on_start=false)\n"
            f"  B                  = disarm\n"
            f"  Hold RB (R1)       = deadman (required for arm motion)\n"
            f"  L-stick            = translate X/Y\n"
            f"  R-stick up/down    = translate Z\n"
            f"  LT / RT            = roll\n"
            f"  D-pad left/right   = yaw\n"
            f"  D-pad up/down      = pitch\n"
            f"  Y                  = gripper fully open  ({self.gripper_open_m:.3f} m)\n"
            f"  X                  = gripper half open   ({self.gripper_half_m:.3f} m)\n"
            f"  A                  = gripper close       ({self.gripper_close_m:.3f} m)\n"
            f"Gripper output: {gripper_output}\n"
            f"Recording: A success / X failure / B discard; Y and HOME disabled.\n"
            f"Publishing arm commands at {rate:.0f} Hz to /rm_driver/movep_canfd_cmd."
        )

    # ------------------------------------------------------------------
    def pose_cb(self, msg: Pose):
        self.workflow.pose(msg)
        pos_zero = (msg.position.x == 0.0
                    and msg.position.y == 0.0
                    and msg.position.z == 0.0)
        quat_invalid = (msg.orientation.x == 0.0
                        and msg.orientation.y == 0.0
                        and msg.orientation.z == 0.0
                        and msg.orientation.w == 0.0)
        if pos_zero or quat_invalid:
            return
        if self.armed or self.workflow.mode != "idle":
            return

        self.tx = msg.position.x
        self.ty = msg.position.y
        self.tz = msg.position.z
        self.qx = msg.orientation.x
        self.qy = msg.orientation.y
        self.qz = msg.orientation.z
        self.qw = msg.orientation.w

        if not self.have_seed:
            self.have_seed = True
            self.get_logger().info(
                f"Seeded from arm: x={self.tx:.4f} y={self.ty:.4f} "
                f"z={self.tz:.4f} qw={self.qw:.4f}"
            )

    # ------------------------------------------------------------------
    def joy_cb(self, msg: Joy):
        if self.workflow.joy(msg):
            self._prev_a = msg.buttons[BTN_A] if len(msg.buttons) > BTN_A else 0
            self._prev_x = msg.buttons[BTN_X] if len(msg.buttons) > BTN_X else 0
            self._prev_y = msg.buttons[BTN_Y] if len(msg.buttons) > BTN_Y else 0
            return

        # --- deadman (RB / R1) ---
        self.deadman = (len(msg.buttons) > BTN_RB and msg.buttons[BTN_RB] == 1)

        # --- HOME button ---
        if self.workflow.allow_home and len(msg.buttons) > BTN_HOME and msg.buttons[BTN_HOME] == 1:
            self.tx = self.home_tx
            self.ty = self.home_ty
            self.tz = self.home_tz

            self.qx = self.home_qx
            self.qy = self.home_qy
            self.qz = self.home_qz
            self.qw = self.home_qw

            self.get_logger().info("Returning to HOME pose.")

        # --- gripper (edge-triggered: fire once per press, no deadman needed) ---
        cur_y = msg.buttons[BTN_Y] if len(msg.buttons) > BTN_Y else 0
        cur_x = msg.buttons[BTN_X] if len(msg.buttons) > BTN_X else 0
        cur_a = msg.buttons[BTN_A] if len(msg.buttons) > BTN_A else 0
        if self.workflow.allow_gripper and cur_y == 1 and self._prev_y == 0:
            self._send_gripper(self.gripper_open_m)
            self.get_logger().info(
                f"Gripper fully open ({self.gripper_open_m:.3f} m)."
            )
        if self.workflow.allow_gripper and cur_x == 1 and self._prev_x == 0:
            self._send_gripper(self.gripper_half_m)
            self.get_logger().info(
                f"Gripper half open ({self.gripper_half_m:.3f} m)."
            )
        if self.workflow.allow_gripper and cur_a == 1 and self._prev_a == 0:
            self._send_gripper(self.gripper_close_m)
            self.get_logger().info(
                f"Gripper closed ({self.gripper_close_m:.3f} m)."
            )
        self._prev_y = cur_y
        self._prev_x = cur_x
        self._prev_a = cur_a

        # --- translation ---
        lx = self._dz(msg.axes[AXIS_LEFT_X])  if len(msg.axes) > AXIS_LEFT_X  else 0.0
        ly = self._dz(msg.axes[AXIS_LEFT_Y])  if len(msg.axes) > AXIS_LEFT_Y  else 0.0
        ry = self._dz(msg.axes[AXIS_RIGHT_Y]) if len(msg.axes) > AXIS_RIGHT_Y else 0.0

        self.vx = ly   # left-stick up    -> +X
        self.vy = lx   # left-stick right -> +Y
        self.vz = ry   # right-stick up   -> +Z

        # --- roll (triggers: 1=idle, -1=full; convert to 0..1) ---
        lt_raw = msg.axes[AXIS_LT] if len(msg.axes) > AXIS_LT else 1.0
        rt_raw = msg.axes[AXIS_RT] if len(msg.axes) > AXIS_RT else 1.0
        lt = (1.0 - lt_raw) / 2.0
        rt = (1.0 - rt_raw) / 2.0
        self.v_roll = rt - lt   # RT -> +roll, LT -> -roll

        # --- yaw (d-pad left/right: +1=left, -1=right) ---
        self.v_yaw = msg.axes[AXIS_DPAD_X] if len(msg.axes) > AXIS_DPAD_X else 0.0

        # --- pitch (d-pad up/down: +1=up, -1=down) ---
        self.v_pitch = msg.axes[AXIS_DPAD_Y] if len(msg.axes) > AXIS_DPAD_Y else 0.0

    def _send_gripper(self, position_m: float):
        if self.gripper_backend == "robot_end":
            msg = Gripperset()
            msg.position = position_m_to_register(
                position_m,
                self.gripper_open_m,
                self.gripper_close_m,
                self.gripper_bridge_open_register,
                self.gripper_bridge_closed_register,
            )
            msg.block = self.gripper_bridge_block
            msg.timeout = self.gripper_bridge_timeout_ms
            self.gripper_bridge_pub.publish(msg)
            return

        msg = Float64MultiArray()
        msg.data = [float(position_m)]
        self.gripper_usb_pub.publish(msg)

    def _dz(self, v: float) -> float:
        return 0.0 if abs(v) < self.dz else v

    def _zero_velocities(self):
        self.vx = self.vy = self.vz = 0.0
        self.v_roll = self.v_pitch = self.v_yaw = 0.0
        self.deadman = False

    # ------------------------------------------------------------------
    def tick(self):
        if self.workflow.tick():
            return
        if not self.armed:
            return

        # Motion only while RB (R1) deadman is held
        if self.deadman:
            # Translate
            self.tx += self.step * self.vx
            self.ty += self.step * self.vy
            self.tz += self.step * self.vz

            # Rotate (world-frame delta quaternions)
            q = (self.qx, self.qy, self.qz, self.qw)

            if abs(self.v_yaw) > 1e-6:
                dq = axis_angle_to_quat(0, 0, 1, self.rot_step * self.v_yaw)
                q = quat_multiply(dq, q)

            if abs(self.v_pitch) > 1e-6:
                dq = axis_angle_to_quat(0, 1, 0, self.rot_step * self.v_pitch)
                q = quat_multiply(dq, q)

            if abs(self.v_roll) > 1e-6:
                dq = axis_angle_to_quat(1, 0, 0, self.rot_step * self.v_roll)
                q = quat_multiply(dq, q)

            self.qx, self.qy, self.qz, self.qw = quat_normalise(q)

        if not self.workflow.position_allowed():
            return
        # Stream to driver (must be continuous before contact).
        msg = Cartepos()
        msg.pose.position.x    = float(self.tx)
        msg.pose.position.y    = float(self.ty)
        msg.pose.position.z    = float(self.tz)
        msg.pose.orientation.x = float(self.qx)
        msg.pose.orientation.y = float(self.qy)
        msg.pose.orientation.z = float(self.qz)
        msg.pose.orientation.w = float(self.qw)
        msg.follow = bool(self.follow_high)
        self.pub.publish(msg)
        self.workflow.published(message=msg)


def main():
    rclpy.init()
    node = XboxXYZTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.workflow.shutdown()
            deadline = monotonic() + 2.5
            while rclpy.ok() and node.workflow.mode == "stopping" and monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.02)
            recorder = node.workflow.recorder
            if recorder is not None:
                recorder.writer.finish(dict(
                    status="incomplete", requested="incomplete", reason="shutdown",
                    stop_confirmed=False, terminal_observation=recorder.terminal,
                    frames=recorder.index, segments=recorder.segment + 1,
                ))
                recorder.writer.thread.join(timeout=5)
        except Exception as error:
            node.get_logger().error(f"Shutdown could not confirm stop/finalization: {error}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
