#!/usr/bin/env python3
"""
Cartesian Record Episodes with PS4 Joystick Control (Teleop Mode)

Extended version that allows arm control BOTH before and during recording.
After recording, arms stay at their current position (use Options to go home if needed).

Control Scheme:

POSITION CONTROL (default - R2 not held):
- Left Stick Y: Move forward/backward (X axis)
- Left Stick X: Move left/right (Y axis)
- Right Stick Y: Move up/down (Z axis)

ORIENTATION CONTROL:
- Right Stick X: Roll (twist around gripper's own axis)
- R2 + Left Stick Y: Pitch (tilt gripper up/down)
- R2 + Left Stick X: Yaw (swing gripper left/right)

OTHER CONTROLS:
- Triangle/X: Open/Close gripper
- L1: Select LEFT arm
- R1 (tap): Select RIGHT arm
- L1+R1: Select BOTH arms
- D-Pad Up/Down: Speed adjustment
- Options: Home pose (when not recording)
- Share: Home pose for both arms (when not recording)

Base Control:
- L2 (hold) + Right Stick: Base movement (overrides arm control)
    - Right Stick Y: Forward/backward
    - Right Stick X: Rotation

Recording:
- Circle: Start/Stop recording
- Square: Discard current recording and restart
- PS Button: Exit program (move to working pose first)
"""

import argparse
from enum import Enum
import os
import signal
import threading
import time

from aloha.constants import (
    DT,
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN,
    FPS,
    IS_MOBILE,
    JOINT_NAMES,
    START_ARM_POSE,
    TASK_CONFIGS,
)
from aloha.real_env import make_real_env
from aloha.robot_utils import (
    get_arm_joint_positions,
    ImageRecorder,
    move_arms,
    move_grippers,
    Recorder,
    setup_follower_bot,
    sleep_arms,
    torque_off,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rclpy
from sensor_msgs.msg import Joy
from tqdm import tqdm

from lerobot_writer import LeRobotWriter, build_aloha_features

# For transformation utilities
from scipy.spatial.transform import Rotation
import modern_robotics as mr


# ============================================================================
# PS4 Controller Mapping
# ============================================================================

class PS4Buttons:
    """PS4 controller button indices."""
    X = 0
    CIRCLE = 1
    TRIANGLE = 2
    SQUARE = 3
    L1 = 4
    R1 = 5
    L2 = 6
    R2 = 7
    SHARE = 8
    OPTIONS = 9
    PS = 10
    L3 = 11
    R3 = 12


class PS4Axes:
    """PS4 controller axis indices."""
    LEFT_STICK_X = 0
    LEFT_STICK_Y = 1
    L2_TRIGGER = 2
    RIGHT_STICK_X = 3
    RIGHT_STICK_Y = 4
    R2_TRIGGER = 5
    DPAD_X = 6
    DPAD_Y = 7


# ============================================================================
# State Enumerations
# ============================================================================

class ArmSelection(Enum):
    """Which arm(s) to control."""
    LEFT = 0
    RIGHT = 1
    BOTH = 2


class RecordingState(Enum):
    """Recording state."""
    IDLE = 0
    RECORDING = 1
    FINISHED = 2


# ============================================================================
# Control Parameters
# ============================================================================

DEADZONE = 0.15
DEFAULT_SPEED_SCALE = 0.5
SPEED_SCALE_MIN = 0.1
SPEED_SCALE_MAX = 1.0
SPEED_SCALE_INCREMENT = 0.1

# Cartesian velocity scales (m/s and rad/s at full stick deflection)
CARTESIAN_LINEAR_SCALE = 0.1   # m/s for position
CARTESIAN_ANGULAR_SCALE = 0.5  # rad/s for orientation

BASE_LINEAR_SCALE = 0.2  # m/s
BASE_ANGULAR_SCALE = 0.4  # rad/s
BASE_LINEAR_ACCEL_LIMIT = 0.15  # m/s² - max linear acceleration for smooth ramping
BASE_ANGULAR_ACCEL_LIMIT = 0.5  # rad/s² - max angular acceleration for smooth ramping

# Exponential moving average filter for base velocity smoothing
# Lower alpha = more smoothing (slower response), higher alpha = less smoothing (faster response)
# 0.1 = very smooth, 0.3 = moderate, 0.5 = light smoothing
BASE_VEL_SMOOTHING_ALPHA = 0.3  # Smoothing factor for EMA filter (lower = more smoothing)

# Command rate reduction - send base commands every N cycles
# 1 = 50Hz, 2 = 25Hz, 3 = ~17Hz, 5 = 10Hz
BASE_CMD_RATE_DIVISOR = 1  # Send commands at full 50Hz rate

# Custom sleep pose [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
# Safer values closer to original config, matches natural resting position
CUSTOM_SLEEP_POSE = [0, -1.91, 1.63, 0, -1.85, 0]

DEBOUNCE_TIME = 0.2  # seconds

# Workspace limits (meters, relative to robot base)
# vx300s specs: 750mm reach, 1500mm total span
# Home position EE: x≈0.26m, z≈0.31m (with START_ARM_POSE)
WORKSPACE_LIMITS = {
    'x': (0.1, 0.65),   # Forward/backward (max reach ~0.75m)
    'y': (-0.4, 0.4), # Left/right
    'z': (None, 0.45),  # Up/down (z_min is distance-dependent, see get_z_min())
}

def get_z_min(x):
    """
    Calculate minimum z based on x position.
    When x is larger (arm extended), z can go lower.
    When x is smaller (arm close), z must stay higher.

    Linear interpolation:
    - x = 0.1m (close) → z_min = 0.20m
    - x = 0.65m (far)  → z_min = -0.05m
    """
    x_min, x_max = 0.1, 0.65
    z_min_at_close = 0.20    # z_min when x is small (close to base)
    z_min_at_far = -0.05     # z_min when x is large (extended)

    # Clamp x to valid range
    x_clamped = np.clip(x, x_min, x_max)

    # Linear interpolation
    t = (x_clamped - x_min) / (x_max - x_min)
    z_min = z_min_at_close + t * (z_min_at_far - z_min_at_close)

    return z_min

# Joint limits for validation (from vx300s URDF, in radians)
# Buffer added to avoid hitting physical limits
JOINT_LIMIT_BUFFER_BIG = 0.349    # ~20 degrees for waist, forearm_roll, wrist_rotate
JOINT_LIMIT_BUFFER_SMALL = 0.087  # ~5 degrees for shoulder, elbow, wrist_angle
JOINT_LIMITS = {
    'waist':        (-3.14 + JOINT_LIMIT_BUFFER_BIG, 3.14 - JOINT_LIMIT_BUFFER_BIG),      # ±180° (with buffer: ±160°)
    'shoulder':     (-1.85 + JOINT_LIMIT_BUFFER_SMALL, 1.26 - JOINT_LIMIT_BUFFER_SMALL),  # -106° to +72° (with buffer: -101° to +67°)
    'elbow':        (-1.76 + JOINT_LIMIT_BUFFER_SMALL, 1.61 - JOINT_LIMIT_BUFFER_SMALL),  # -101° to +92° (with buffer: -96° to +87°)
    'forearm_roll': (-3.14 + JOINT_LIMIT_BUFFER_BIG, 3.14 - JOINT_LIMIT_BUFFER_BIG),      # ±180° (with buffer: ±160°)
    'wrist_angle':  (-1.87 + JOINT_LIMIT_BUFFER_SMALL, 2.23 - JOINT_LIMIT_BUFFER_SMALL),  # -107° to +128° (with buffer: -102° to +123°)
    'wrist_rotate': (-3.14 + JOINT_LIMIT_BUFFER_BIG, 3.14 - JOINT_LIMIT_BUFFER_BIG),      # ±180° (with buffer: ±160°)
}

# IK fallback parameters
IK_FALLBACK_FACTOR = 0.3     # How much to step back toward last good pose on IK failure
IK_MAX_CONSECUTIVE_FAILS = 5 # After this many fails, force fallback


# ============================================================================
# Helper Functions
# ============================================================================

def pose_matrix_to_components(T):
    """
    Extract position and euler angles from a 4x4 transformation matrix.
    Returns: (x, y, z, roll, pitch, yaw)
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rotation = Rotation.from_matrix(T[:3, :3])
    roll, pitch, yaw = rotation.as_euler('xyz')
    return x, y, z, roll, pitch, yaw


def components_to_pose_matrix(x, y, z, roll, pitch, yaw):
    """
    Create a 4x4 transformation matrix from position and euler angles.
    """
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
    T[:3, :3] = rotation.as_matrix()
    return T


# ============================================================================
# Cartesian Joystick Recording Controller (Teleop Mode)
# ============================================================================

class CartesianRecorderTeleop:
    """Cartesian PS4 joystick controller for recording ALOHA episodes with teleop before recording."""

    def __init__(self, node, follower_left, follower_right, env, base=None):
        self.node = node
        self.follower_left = follower_left
        self.follower_right = follower_right
        self.env = env
        self.base = base

        # Control state
        self.arm_selection = ArmSelection.BOTH
        self.torque_enabled = True
        self.speed_scale = DEFAULT_SPEED_SCALE
        self.pending_action = None  # Set by callbacks, handled by main loop

        # Recording state
        self.recording_state = RecordingState.IDLE
        self.should_discard = False
        self.exit_requested = False  # For PS button exit

        # Current end-effector poses (x, y, z, roll, pitch, yaw)
        self._init_ee_poses()

        # Last known good poses for IK fallback (initialized after _init_ee_poses)
        self.last_good_left_pose = self.left_ee_pose.copy()
        self.last_good_right_pose = self.right_ee_pose.copy()

        # Current joint positions (for action output)
        self.current_left_joints = np.array(get_arm_joint_positions(follower_left))
        self.current_right_joints = np.array(get_arm_joint_positions(follower_right))
        self.current_left_gripper = FOLLOWER_GRIPPER_JOINT_CLOSE
        self.current_right_gripper = FOLLOWER_GRIPPER_JOINT_CLOSE

        # Base velocity command (for recording)
        self.current_base_action = np.array([0.0, 0.0])
        # Base velocity ramping (for smooth acceleration)
        self.last_base_linear_vel = 0.0
        self.last_base_angular_vel = 0.0
        # EMA filtered velocities (for additional smoothing)
        self.filtered_base_linear_vel = 0.0
        self.filtered_base_angular_vel = 0.0
        # Command rate reduction counter
        self.base_cmd_counter = 0

        # Joystick state
        self.last_joy_msg = None
        self.joy_connected = False

        # Button timing for debounce
        self.button_press_time = {}
        self.button_last_state = {}

        # PS button hold detection for exit
        self.ps_button_hold_start = None
        self.PS_HOLD_EXIT_TIME = 2.0  # Hold PS button for 2 seconds to exit

        # IK failure tracking
        self.ik_fail_count = 0

        # Workspace limit hit tracking (to avoid spamming warnings)
        self.limit_hit_count = 0

        # Subscribe to joystick
        self.joy_sub = node.create_subscription(
            Joy,
            '/mobile_base/joy',
            self.joy_callback,
            10
        )

        self.print_status()

    def _init_ee_poses(self):
        """Initialize end-effector poses from current robot state."""
        try:
            # Get current EE pose from FK
            T_left = self.follower_left.arm.get_ee_pose()
            T_right = self.follower_right.arm.get_ee_pose()

            self.left_ee_pose = list(pose_matrix_to_components(T_left))
            self.right_ee_pose = list(pose_matrix_to_components(T_right))

            print(f'Left EE: x={self.left_ee_pose[0]:.3f}, y={self.left_ee_pose[1]:.3f}, z={self.left_ee_pose[2]:.3f}')
            print(f'Right EE: x={self.right_ee_pose[0]:.3f}, y={self.right_ee_pose[1]:.3f}, z={self.right_ee_pose[2]:.3f}')
        except Exception as e:
            print(f'Warning: Could not get EE pose, using defaults: {e}')
            # Default pose (roughly home position EE)
            self.left_ee_pose = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
            self.right_ee_pose = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]

    def joy_callback(self, msg: Joy):
        """Process incoming joystick messages."""
        if not self.joy_connected:
            self.joy_connected = True
            print('\n>>> PS4 controller connected!')
            self.print_controls()

        self.last_joy_msg = msg
        self._process_buttons(msg)

    def _process_buttons(self, msg: Joy):
        """Process button presses for state changes."""
        current_time = time.time()

        # PS button hold detection for exit
        ps_pressed = msg.buttons[PS4Buttons.PS] if len(msg.buttons) > PS4Buttons.PS else False
        if ps_pressed:
            if self.ps_button_hold_start is None:
                self.ps_button_hold_start = current_time
            elif current_time - self.ps_button_hold_start >= self.PS_HOLD_EXIT_TIME:
                print('\n>>> PS button held - EXIT REQUESTED')
                self.exit_requested = True
        else:
            self.ps_button_hold_start = None

        # Recording control (Circle button)
        if self._detect_button_tap(PS4Buttons.CIRCLE, msg, current_time):
            if self.recording_state == RecordingState.IDLE:
                self.recording_state = RecordingState.RECORDING
                print('\n>>> RECORDING STARTED! Press Circle to stop.')
            elif self.recording_state == RecordingState.RECORDING:
                self.recording_state = RecordingState.FINISHED
                print('\n>>> RECORDING STOPPED!')

        # Discard recording (Square button)
        if self._detect_button_tap(PS4Buttons.SQUARE, msg, current_time):
            if self.recording_state == RecordingState.RECORDING:
                self.should_discard = True
                self.recording_state = RecordingState.IDLE
                print('\n>>> RECORDING DISCARDED!')

        # Home pose (Options button) - active arm(s)
        if self._detect_button_tap(PS4Buttons.OPTIONS, msg, current_time):
            if self.recording_state != RecordingState.RECORDING:
                self.pending_action = 'home_active'

        # Home pose (Share button) - both arms
        if self._detect_button_tap(PS4Buttons.SHARE, msg, current_time):
            if self.recording_state != RecordingState.RECORDING:
                self.pending_action = 'home_both'

        # Arm selection
        l1_pressed = msg.buttons[PS4Buttons.L1] if len(msg.buttons) > PS4Buttons.L1 else False
        r1_pressed = msg.buttons[PS4Buttons.R1] if len(msg.buttons) > PS4Buttons.R1 else False

        if l1_pressed and r1_pressed:
            if self.arm_selection != ArmSelection.BOTH:
                self.arm_selection = ArmSelection.BOTH
                print('\n>>> Controlling: BOTH arms')
                self.print_status()
        elif self._detect_button_tap(PS4Buttons.L1, msg, current_time):
            if self.arm_selection != ArmSelection.LEFT:
                self.arm_selection = ArmSelection.LEFT
                print('\n>>> Controlling: LEFT arm')
                self.print_status()
        elif self._detect_button_tap(PS4Buttons.R1, msg, current_time):
            if self.arm_selection != ArmSelection.RIGHT:
                self.arm_selection = ArmSelection.RIGHT
                print('\n>>> Controlling: RIGHT arm')
                self.print_status()

        # Speed control (D-Pad Up/Down)
        if len(msg.axes) > PS4Axes.DPAD_Y:
            dpad_y = msg.axes[PS4Axes.DPAD_Y]
            if dpad_y > 0.5:  # Up
                if self._detect_dpad_tap('dpad_up', True, current_time):
                    self.speed_scale = min(SPEED_SCALE_MAX, self.speed_scale + SPEED_SCALE_INCREMENT)
                    print(f'\n>>> Speed: {self.speed_scale:.0%}')
            elif dpad_y < -0.5:  # Down
                if self._detect_dpad_tap('dpad_down', True, current_time):
                    self.speed_scale = max(SPEED_SCALE_MIN, self.speed_scale - SPEED_SCALE_INCREMENT)
                    print(f'\n>>> Speed: {self.speed_scale:.0%}')
            else:
                self.button_last_state['dpad_up'] = False
                self.button_last_state['dpad_down'] = False

    def _detect_button_tap(self, button_idx: int, msg: Joy, current_time: float) -> bool:
        """Detect a button tap (rising edge with debounce)."""
        if len(msg.buttons) <= button_idx:
            return False

        pressed = msg.buttons[button_idx]
        last_state = self.button_last_state.get(button_idx, False)
        last_time = self.button_press_time.get(button_idx, 0)

        if pressed and not last_state and (current_time - last_time) > DEBOUNCE_TIME:
            self.button_press_time[button_idx] = current_time
            self.button_last_state[button_idx] = True
            return True

        if not pressed:
            self.button_last_state[button_idx] = False

        return False

    def _detect_dpad_tap(self, key: str, pressed: bool, current_time: float) -> bool:
        """Detect a D-Pad tap (rising edge with debounce)."""
        last_state = self.button_last_state.get(key, False)
        last_time = self.button_press_time.get(key, 0)

        if pressed and not last_state and (current_time - last_time) > DEBOUNCE_TIME:
            self.button_press_time[key] = current_time
            self.button_last_state[key] = True
            return True

        return False

    def update(self, is_recording=False):
        """
        Main update loop - call at 50Hz.

        Args:
            is_recording: If True, compute IK without publishing (env.step handles it).
                         If False (IDLE/teleop mode), publish commands directly.
        Returns:
            The action taken.
        """
        if self.last_joy_msg is None:
            # Still hold arm and gripper positions even without joystick input
            if not is_recording:
                self.follower_left.arm.set_joint_positions(
                    list(self.current_left_joints), blocking=False)
                self.follower_right.arm.set_joint_positions(
                    list(self.current_right_joints), blocking=False)
                self._send_gripper_command(self.follower_left, self.current_left_gripper)
                self._send_gripper_command(self.follower_right, self.current_right_gripper)
            return self.get_current_action()

        msg = self.last_joy_msg

        # Update arm control - mode depends on recording state
        self._update_arm_control_cartesian(msg, publish_commands=not is_recording)
        self._update_base_control(msg)

        return self.get_current_action()

    def get_current_action(self):
        """Get current action in the format expected by the environment."""
        # Action format: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        action = np.zeros(14)

        # Left arm
        action[:6] = self.current_left_joints
        # Left gripper (normalized)
        action[6] = self._normalize_gripper(self.current_left_gripper)

        # Right arm
        action[7:13] = self.current_right_joints
        # Right gripper (normalized)
        action[13] = self._normalize_gripper(self.current_right_gripper)

        return action

    def _normalize_gripper(self, gripper_joint):
        """Normalize gripper joint position to [0, 1]."""
        return (gripper_joint - FOLLOWER_GRIPPER_JOINT_CLOSE) / (
            FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE
        )

    def _update_arm_control_cartesian(self, msg: Joy, publish_commands=True):
        """Update arm positions using Cartesian control scheme.

        Control Mapping:
        - Left Stick X/Y: Position X/Y (when R2 NOT held)
        - R2 + Left Stick Y: Pitch (tilt up/down)
        - R2 + Left Stick X: Yaw (swing left/right)
        - Right Stick Y: Position Z (up/down)
        - Right Stick X: Roll (twist around own axis)
        - L2 + Right Stick: Base control (overrides arm)

        Args:
            publish_commands: If True, publish joint commands directly (teleop mode).
                            If False, only compute IK (recording mode, env.step publishes).
        """
        if len(msg.axes) <= PS4Axes.RIGHT_STICK_Y:
            return

        # Check modifier buttons
        l2_pressed = msg.buttons[PS4Buttons.L2] if len(msg.buttons) > PS4Buttons.L2 else False
        r2_pressed = msg.buttons[PS4Buttons.R2] if len(msg.buttons) > PS4Buttons.R2 else False

        # Get stick values
        left_x = msg.axes[PS4Axes.LEFT_STICK_X]
        left_y = msg.axes[PS4Axes.LEFT_STICK_Y]
        right_x = msg.axes[PS4Axes.RIGHT_STICK_X] if not l2_pressed else 0.0
        right_y = msg.axes[PS4Axes.RIGHT_STICK_Y] if not l2_pressed else 0.0

        # Apply deadzone
        left_x = left_x if abs(left_x) > DEADZONE else 0.0
        left_y = left_y if abs(left_y) > DEADZONE else 0.0
        right_x = right_x if abs(right_x) > DEADZONE else 0.0
        right_y = right_y if abs(right_y) > DEADZONE else 0.0

        # Calculate deltas
        linear_scale = self.speed_scale * DT * CARTESIAN_LINEAR_SCALE
        angular_scale = self.speed_scale * DT * CARTESIAN_ANGULAR_SCALE

        # Position and orientation based on R2 modifier
        if r2_pressed:
            # R2 held: Left stick controls orientation (pitch/yaw)
            dx = 0.0
            dy = 0.0
            dpitch = left_y * angular_scale   # Left Stick Y: Pitch (tilt up/down)
            dyaw = left_x * angular_scale     # Left Stick X: Yaw (swing left/right)
        else:
            # R2 not held: Left stick controls position (X/Y)
            dx = left_y * linear_scale        # Left Stick Y: Forward/backward (X)
            dy = left_x * linear_scale        # Left Stick X: Left/right (Y)
            dpitch = 0.0
            dyaw = 0.0

        # Right stick always controls Z and Roll (unless L2 for base)
        dz = right_y * linear_scale           # Right Stick Y: Up/down (Z)
        droll = right_x * angular_scale       # Right Stick X: Roll (twist)

        # Update end-effector poses and send commands
        if self.arm_selection in (ArmSelection.LEFT, ArmSelection.BOTH):
            self._update_single_arm_cartesian(
                self.follower_left,
                self.left_ee_pose,
                dx, dy, dz, droll, dpitch, dyaw,
                'left',
                publish_commands
            )
        elif publish_commands:
            # Hold non-active arm in position to prevent sag
            self.follower_left.arm.set_joint_positions(
                list(self.current_left_joints), blocking=False)

        if self.arm_selection in (ArmSelection.RIGHT, ArmSelection.BOTH):
            self._update_single_arm_cartesian(
                self.follower_right,
                self.right_ee_pose,
                dx, dy, dz, droll, dpitch, dyaw,
                'right',
                publish_commands
            )
        elif publish_commands:
            # Hold non-active arm in position to prevent sag
            self.follower_right.arm.set_joint_positions(
                list(self.current_right_joints), blocking=False)

        # Gripper control via Triangle (open) / X (close)
        gripper_delta = 0.0
        if len(msg.buttons) > PS4Buttons.TRIANGLE and msg.buttons[PS4Buttons.TRIANGLE]:
            gripper_delta = 0.05  # Open
        elif len(msg.buttons) > PS4Buttons.X and msg.buttons[PS4Buttons.X]:
            gripper_delta = -0.05  # Close

        if gripper_delta != 0.0:
            if self.arm_selection in (ArmSelection.LEFT, ArmSelection.BOTH):
                self.current_left_gripper += gripper_delta
                self.current_left_gripper = np.clip(
                    self.current_left_gripper,
                    FOLLOWER_GRIPPER_JOINT_CLOSE,
                    FOLLOWER_GRIPPER_JOINT_OPEN
                )
                self._send_gripper_command(self.follower_left, self.current_left_gripper)

            if self.arm_selection in (ArmSelection.RIGHT, ArmSelection.BOTH):
                self.current_right_gripper += gripper_delta
                self.current_right_gripper = np.clip(
                    self.current_right_gripper,
                    FOLLOWER_GRIPPER_JOINT_CLOSE,
                    FOLLOWER_GRIPPER_JOINT_OPEN
                )
                self._send_gripper_command(self.follower_right, self.current_right_gripper)

    def _update_single_arm_cartesian(self, bot, ee_pose, dx, dy, dz, droll, dpitch, dyaw, arm_name, publish_commands=True):
        """Update a single arm using Cartesian control with IK.

        Args:
            publish_commands: If True, publish joint commands directly.
                            If False, only compute IK and update tracked positions.
        """
        # Get last good pose and current joints for this arm
        if arm_name == 'left':
            last_good_pose = self.last_good_left_pose
            current_joints = self.current_left_joints
        else:
            last_good_pose = self.last_good_right_pose
            current_joints = self.current_right_joints

        # If no movement requested, just hold current position
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6 and \
           abs(droll) < 1e-6 and abs(dpitch) < 1e-6 and abs(dyaw) < 1e-6:
            if publish_commands:
                bot.arm.set_joint_positions(list(current_joints), blocking=False)
            return

        # Calculate new target pose
        new_x = ee_pose[0] + dx
        new_y = ee_pose[1] + dy
        new_z = ee_pose[2] + dz
        new_roll = ee_pose[3] + droll
        new_pitch = ee_pose[4] + dpitch
        new_yaw = ee_pose[5] + dyaw

        # Store pre-limit values to detect if limits were hit
        pre_limit_x, pre_limit_y, pre_limit_z = new_x, new_y, new_z

        # Apply rectangular workspace limits (box)
        new_x = np.clip(new_x, WORKSPACE_LIMITS['x'][0], WORKSPACE_LIMITS['x'][1])
        new_y = np.clip(new_y, WORKSPACE_LIMITS['y'][0], WORKSPACE_LIMITS['y'][1])

        # Apply distance-dependent z limit (z_min depends on x position)
        z_min = get_z_min(new_x)
        z_max = WORKSPACE_LIMITS['z'][1]
        new_z = np.clip(new_z, z_min, z_max)

        # Check if box limits were hit
        box_limit_hit = (new_x != pre_limit_x or new_y != pre_limit_y or new_z != pre_limit_z)

        # Print limit warnings
        if box_limit_hit:
            self.limit_hit_count += 1
            if self.limit_hit_count == 1:
                print(f'\n>>> {arm_name.upper()} arm hit workspace limit', flush=True)
            elif self.limit_hit_count % 50 == 0:
                print(f'>>> Workspace limit still active ({self.limit_hit_count} hits)', flush=True)
        else:
            self.limit_hit_count = 0

        # Wrap angles to [-pi, pi]
        new_roll = np.arctan2(np.sin(new_roll), np.cos(new_roll))
        new_pitch = np.arctan2(np.sin(new_pitch), np.cos(new_pitch))
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

        try:
            # Build target transformation matrix
            T_target = self._build_pose_matrix(new_x, new_y, new_z, new_roll, new_pitch, new_yaw)

            # Compute IK using Modern Robotics
            theta_list, success = mr.IKinSpace(
                Slist=bot.arm.robot_des.Slist,
                M=bot.arm.robot_des.M,
                T=T_target,
                thetalist0=current_joints,
                eomg=0.01,  # Orientation tolerance
                ev=0.001,   # Position tolerance (1mm)
            )

            # Validate IK solution against joint limits
            if success:
                joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
                for i, (name, angle) in enumerate(zip(joint_names, theta_list)):
                    limits = JOINT_LIMITS[name]
                    if angle < limits[0] or angle > limits[1]:
                        success = False
                        if self.ik_fail_count == 0:
                            print(f'\n>>> {arm_name.upper()} arm: IK solution exceeds {name} joint limit '
                                  f'({np.degrees(angle):.1f}° not in [{np.degrees(limits[0]):.0f}°, {np.degrees(limits[1]):.0f}°])',
                                  flush=True)
                        break

            if success:
                # Update stored pose
                ee_pose[0] = new_x
                ee_pose[1] = new_y
                ee_pose[2] = new_z
                ee_pose[3] = new_roll
                ee_pose[4] = new_pitch
                ee_pose[5] = new_yaw

                # Update last good pose
                if arm_name == 'left':
                    self.last_good_left_pose = [new_x, new_y, new_z, new_roll, new_pitch, new_yaw]
                    self.current_left_joints = np.array(theta_list)
                else:
                    self.last_good_right_pose = [new_x, new_y, new_z, new_roll, new_pitch, new_yaw]
                    self.current_right_joints = np.array(theta_list)

                # Publish commands directly in teleop mode (not recording)
                if publish_commands:
                    bot.arm.set_joint_positions(list(theta_list), blocking=False)

                self.ik_fail_count = 0
            else:
                # IK failed - apply fallback
                self.ik_fail_count += 1

                if self.ik_fail_count >= IK_MAX_CONSECUTIVE_FAILS:
                    # Fallback: move target back toward last known good pose
                    ee_pose[0] += IK_FALLBACK_FACTOR * (last_good_pose[0] - new_x)
                    ee_pose[1] += IK_FALLBACK_FACTOR * (last_good_pose[1] - new_y)
                    ee_pose[2] += IK_FALLBACK_FACTOR * (last_good_pose[2] - new_z)
                    ee_pose[3] += IK_FALLBACK_FACTOR * (last_good_pose[3] - new_roll)
                    ee_pose[4] += IK_FALLBACK_FACTOR * (last_good_pose[4] - new_pitch)
                    ee_pose[5] += IK_FALLBACK_FACTOR * (last_good_pose[5] - new_yaw)

                # Print warnings (flush=True ensures immediate output)
                if self.ik_fail_count == 1:
                    # First failure - print with newline for visibility
                    print(f'\n>>> IK failed for {arm_name} arm (target out of reach)', flush=True)
                elif self.ik_fail_count == IK_MAX_CONSECUTIVE_FAILS:
                    # Fallback triggered
                    print(f'>>> IK fallback: {arm_name} arm moving back toward reachable zone', flush=True)
                elif self.ik_fail_count % 25 == 0:
                    # Periodic warning during ongoing failures
                    print(f'>>> IK still failing for {arm_name} arm ({self.ik_fail_count} consecutive failures)', flush=True)

        except Exception as e:
            self.ik_fail_count += 1
            if self.ik_fail_count == 1:
                print(f'\n>>> IK error for {arm_name} arm: {e}', flush=True)

    def _build_pose_matrix(self, x, y, z, roll, pitch, yaw):
        """Build a 4x4 transformation matrix from position and euler angles."""
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = x, y, z
        rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
        T[:3, :3] = rotation.as_matrix()
        return T

    def _send_gripper_command(self, bot, gripper_pos):
        """Send gripper command."""
        gripper_cmd = JointSingleCommand(name='gripper')
        gripper_cmd.cmd = gripper_pos
        bot.gripper.core.pub_single.publish(gripper_cmd)

    def _update_base_control(self, msg: Joy):
        """Update base velocity based on joystick input with smooth ramping."""
        if self.base is None:
            self.current_base_action = np.array([0.0, 0.0])
            return

        if len(msg.axes) <= PS4Axes.RIGHT_STICK_Y:
            return

        # Check L2 safety enable
        l2_pressed = msg.buttons[PS4Buttons.L2] if len(msg.buttons) > PS4Buttons.L2 else False

        if l2_pressed:
            # Right stick for base control
            target_linear = msg.axes[PS4Axes.RIGHT_STICK_Y]
            target_angular = msg.axes[PS4Axes.RIGHT_STICK_X]

            # Apply deadzone
            if abs(target_linear) < DEADZONE:
                target_linear = 0.0
            if abs(target_angular) < DEADZONE:
                target_angular = 0.0

            # Scale to target velocities
            target_linear *= BASE_LINEAR_SCALE * self.speed_scale
            target_angular *= BASE_ANGULAR_SCALE * self.speed_scale
        else:
            # Stop base when L2 not held
            target_linear = 0.0
            target_angular = 0.0

        # Apply acceleration limiting for smooth ramping
        max_linear_delta = BASE_LINEAR_ACCEL_LIMIT * DT
        max_angular_delta = BASE_ANGULAR_ACCEL_LIMIT * DT

        # Ramp linear velocity
        linear_diff = target_linear - self.last_base_linear_vel
        if abs(linear_diff) > max_linear_delta:
            linear_vel = self.last_base_linear_vel + np.sign(linear_diff) * max_linear_delta
        else:
            linear_vel = target_linear

        # Ramp angular velocity
        angular_diff = target_angular - self.last_base_angular_vel
        if abs(angular_diff) > max_angular_delta:
            angular_vel = self.last_base_angular_vel + np.sign(angular_diff) * max_angular_delta
        else:
            angular_vel = target_angular

        # Update last velocities for next iteration
        self.last_base_linear_vel = linear_vel
        self.last_base_angular_vel = angular_vel

        # EMA filter disabled - use ramped velocities directly
        # # Apply EMA filter for additional smoothing
        # # EMA: filtered = alpha * new + (1 - alpha) * filtered_prev
        # alpha = BASE_VEL_SMOOTHING_ALPHA
        # self.filtered_base_linear_vel = alpha * linear_vel + (1 - alpha) * self.filtered_base_linear_vel
        # self.filtered_base_angular_vel = alpha * angular_vel + (1 - alpha) * self.filtered_base_angular_vel

        # Store action for recording (always update)
        self.current_base_action = np.array([linear_vel, angular_vel])

        # Command rate reduction - only send to base every N cycles
        self.base_cmd_counter += 1
        if self.base_cmd_counter >= BASE_CMD_RATE_DIVISOR:
            self.base_cmd_counter = 0
            self.base.base.command_velocity_xyaw(x=linear_vel, yaw=angular_vel)

    def _go_to_home_pose(self):
        """Move active arm(s) to home pose based on current arm selection."""
        start_arm_qpos = START_ARM_POSE[:6]

        if self.arm_selection == ArmSelection.LEFT:
            print('>>> Moving LEFT arm to HOME...')
            move_arms([self.follower_left], [start_arm_qpos], moving_time=4.0)
            self.current_left_joints = np.array(start_arm_qpos)
            T_left = mr.FKinSpace(
                self.follower_left.arm.robot_des.M,
                self.follower_left.arm.robot_des.Slist,
                start_arm_qpos
            )
            self.left_ee_pose = list(pose_matrix_to_components(T_left))
            self.last_good_left_pose = self.left_ee_pose.copy()
        elif self.arm_selection == ArmSelection.RIGHT:
            print('>>> Moving RIGHT arm to HOME...')
            move_arms([self.follower_right], [start_arm_qpos], moving_time=4.0)
            self.current_right_joints = np.array(start_arm_qpos)
            T_right = mr.FKinSpace(
                self.follower_right.arm.robot_des.M,
                self.follower_right.arm.robot_des.Slist,
                start_arm_qpos
            )
            self.right_ee_pose = list(pose_matrix_to_components(T_right))
            self.last_good_right_pose = self.right_ee_pose.copy()
        else:
            print('>>> Moving BOTH arms to HOME...')
            move_arms(
                [self.follower_left, self.follower_right],
                [start_arm_qpos, start_arm_qpos],
                moving_time=4.0
            )
            self.current_left_joints = np.array(start_arm_qpos)
            self.current_right_joints = np.array(start_arm_qpos)
            T_left = mr.FKinSpace(
                self.follower_left.arm.robot_des.M,
                self.follower_left.arm.robot_des.Slist,
                start_arm_qpos
            )
            T_right = mr.FKinSpace(
                self.follower_right.arm.robot_des.M,
                self.follower_right.arm.robot_des.Slist,
                start_arm_qpos
            )
            self.left_ee_pose = list(pose_matrix_to_components(T_left))
            self.right_ee_pose = list(pose_matrix_to_components(T_right))
            self.last_good_left_pose = self.left_ee_pose.copy()
            self.last_good_right_pose = self.right_ee_pose.copy()

        # Reset IK failure state
        self.ik_fail_count = 0
        self.limit_hit_count = 0
        # Clear stale joystick message to prevent immediate movement
        self.last_joy_msg = None

    def _go_to_home_pose_both(self):
        """Move BOTH arms to home pose (ignores arm_selection)."""
        start_arm_qpos = START_ARM_POSE[:6]
        print('>>> Moving BOTH arms to HOME...')
        move_arms(
            [self.follower_left, self.follower_right],
            [start_arm_qpos, start_arm_qpos],
            moving_time=4.0
        )
        self.current_left_joints = np.array(start_arm_qpos)
        self.current_right_joints = np.array(start_arm_qpos)
        T_left = mr.FKinSpace(
            self.follower_left.arm.robot_des.M,
            self.follower_left.arm.robot_des.Slist,
            start_arm_qpos
        )
        T_right = mr.FKinSpace(
            self.follower_right.arm.robot_des.M,
            self.follower_right.arm.robot_des.Slist,
            start_arm_qpos
        )
        self.left_ee_pose = list(pose_matrix_to_components(T_left))
        self.right_ee_pose = list(pose_matrix_to_components(T_right))
        self.last_good_left_pose = self.left_ee_pose.copy()
        self.last_good_right_pose = self.right_ee_pose.copy()
        self.ik_fail_count = 0
        self.limit_hit_count = 0
        self.last_joy_msg = None

    def _go_to_sleep_pose(self):
        """Move arms to sleep pose."""
        # Use custom sleep pose instead of SDK config
        sleep_left = list(CUSTOM_SLEEP_POSE)
        sleep_right = list(CUSTOM_SLEEP_POSE)
        print(f'>>> Using CUSTOM sleep pose: {CUSTOM_SLEEP_POSE}')

        # Move to home first, then to custom sleep pose
        bots = [self.follower_left, self.follower_right]
        print('>>> Moving to HOME pose first...')
        move_arms(bots, [[0.0, -0.96, 1.16, 0.0, -0.3, 0.0]] * len(bots), moving_time=4.0)
        print('>>> HOME reached. Now moving to CUSTOM SLEEP pose...')
        move_arms(bots, [sleep_left, sleep_right], moving_time=4.0)
        time.sleep(0.5)  # Wait for position feedback to update
        print('>>> CUSTOM SLEEP pose reached.')

        # Update tracked joint positions
        self.current_left_joints = np.array(sleep_left)
        self.current_right_joints = np.array(sleep_right)

        # Compute EE poses directly from commanded joints using FK
        T_left = mr.FKinSpace(
            self.follower_left.arm.robot_des.M,
            self.follower_left.arm.robot_des.Slist,
            sleep_left
        )
        T_right = mr.FKinSpace(
            self.follower_right.arm.robot_des.M,
            self.follower_right.arm.robot_des.Slist,
            sleep_right
        )
        self.left_ee_pose = list(pose_matrix_to_components(T_left))
        self.right_ee_pose = list(pose_matrix_to_components(T_right))

        print(f'Sleep EE: x={self.left_ee_pose[0]:.3f}, y={self.left_ee_pose[1]:.3f}, z={self.left_ee_pose[2]:.3f}')

        # Update last good poses
        self.last_good_left_pose = self.left_ee_pose.copy()
        self.last_good_right_pose = self.right_ee_pose.copy()
        # Reset IK failure state
        self.ik_fail_count = 0
        self.limit_hit_count = 0
        # Clear stale joystick message to prevent immediate movement
        self.last_joy_msg = None

    def reset_for_new_episode(self):
        """Reset state for recording a new episode."""
        self.recording_state = RecordingState.IDLE
        self.should_discard = False

        # Read current joint positions from robot
        left_joints = list(get_arm_joint_positions(self.follower_left))
        right_joints = list(get_arm_joint_positions(self.follower_right))

        self.current_left_joints = np.array(left_joints)
        self.current_right_joints = np.array(right_joints)

        # Compute EE poses from the joint positions we just read (ensures consistency)
        T_left = mr.FKinSpace(
            self.follower_left.arm.robot_des.M,
            self.follower_left.arm.robot_des.Slist,
            left_joints
        )
        T_right = mr.FKinSpace(
            self.follower_right.arm.robot_des.M,
            self.follower_right.arm.robot_des.Slist,
            right_joints
        )
        self.left_ee_pose = list(pose_matrix_to_components(T_left))
        self.right_ee_pose = list(pose_matrix_to_components(T_right))

        # Update last good poses
        self.last_good_left_pose = self.left_ee_pose.copy()
        self.last_good_right_pose = self.right_ee_pose.copy()
        # Reset IK failure state
        self.ik_fail_count = 0
        self.limit_hit_count = 0

    def print_status(self):
        """Print current status to console."""
        arm_str = ['LEFT', 'RIGHT', 'BOTH'][self.arm_selection.value]
        rec_str = self.recording_state.name

        print(f'\rArm: {arm_str} | Speed: {self.speed_scale:.0%} | Recording: {rec_str} | Mode: CARTESIAN', end='    \n')

    def print_controls(self):
        """Print control instructions."""
        print('\n' + '=' * 60)
        print('CARTESIAN PS4 RECORDER (TELEOP MODE) - ALOHA')
        print('=' * 60)
        print('\nRECORDING CONTROLS:')
        print('  Circle       : Start/Stop recording')
        print('  Square       : Discard recording')
        print('  PS (hold 2s) : Exit program (return to working pose)')
        print('\nSYSTEM CONTROLS:')
        print('  Options      : Home pose (when not recording)')
        print('  Share        : Home pose - both arms (when not recording)')
        print('  D-Pad U/D    : Speed adjustment')
        print('\nPOSITION CONTROL (default):')
        print('  Left Stick Y : Move forward/backward (X)')
        print('  Left Stick X : Move left/right (Y)')
        print('  Right Stick Y: Move up/down (Z)')
        print('\nORIENTATION CONTROL:')
        print('  Right Stick X: Roll (twist around own axis)')
        print('  R2 + Left Stick Y: Pitch (tilt up/down)')
        print('  R2 + Left Stick X: Yaw (swing left/right)')
        print('\nGRIPPER:')
        print('  Triangle     : Open gripper')
        print('  X            : Close gripper')
        print('\nARM SELECTION:')
        print('  L1           : Select LEFT arm')
        print('  R1 (tap)     : Select RIGHT arm')
        print('  L1+R1        : Select BOTH arms')
        print('\nBASE CONTROL:')
        print('  L2 (hold)    : Enable base (overrides arm)')
        print('  + Right Stick Y: Forward/backward')
        print('  + Right Stick X: Rotation')
        print('=' * 60)
        print('\n*** TELEOP MODE: Control arms before AND during recording ***\n')


# ============================================================================
# Recording Functions
# ============================================================================

def opening_ceremony(follower_left: InterbotixManipulatorXS,
                     follower_right: InterbotixManipulatorXS) -> None:
    """Initialize follower arms for joystick recording."""
    print('Initializing follower arms...')

    setup_follower_bot(follower_left)
    setup_follower_bot(follower_right)

    # Move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    print('Moving to start position...')
    move_arms(
        [follower_left, follower_right],
        [start_arm_qpos, start_arm_qpos],
        moving_time=4.0,
    )

    # Move grippers to closed position
    move_grippers(
        [follower_left, follower_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE, FOLLOWER_GRIPPER_JOINT_CLOSE],
        moving_time=0.5
    )

    print('Initialization complete!')


def capture_one_episode(
    recorder,
    env,
    max_timesteps,
    camera_names,
    lerobot_writer,
    task_name=None,
):
    """Capture a single episode with teleop control before and during recording."""

    print(f'\n{"="*60}')
    #print(f'Ready to record: {dataset_name}')
    print(f'Ready to record: {task_name}')
    print(f'{"="*60}')
    print('\nTELEOP MODE ACTIVE - Control arms now!')
    print('Press Circle to start recording when ready.')
    print('Press PS button (hold 2s) to exit program.\n')

    # TELEOP MODE: Allow control before recording starts
    while rclpy.ok() and recorder.recording_state == RecordingState.IDLE:
        # Update with publish_commands=True (teleop mode)
        recorder.update(is_recording=False)

        # Handle pending pose actions in main thread (gapless transition)
        if recorder.pending_action == 'home_active':
            recorder._go_to_home_pose()
            recorder.pending_action = None
        elif recorder.pending_action == 'home_both':
            recorder._go_to_home_pose_both()
            recorder.pending_action = None

        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

        # Check for exit request
        if recorder.exit_requested:
            return False, True  # (success, exit_requested)

    if not rclpy.ok():
        return False, False

    # Hold current position during transition to prevent sag
    hold_action = recorder.get_current_action()
    env.step(hold_action, get_base_vel=False)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    base_actions = []
    actual_dt_history = []
    time0 = time.time()
    DT_RECORD = 1 / FPS

    print(f'\nRecording up to {max_timesteps} timesteps...')
    print('Press Circle to stop recording early.')

    for t in tqdm(range(max_timesteps)):
        if recorder.should_discard:
            print('\nRecording discarded!')
            break
        if recorder.recording_state != RecordingState.RECORDING:
            break

        t0 = time.time()

        # Get action from joystick (with is_recording=True, so IK computes but doesn't publish)
        action = recorder.update(is_recording=True)
        t1 = time.time()

        # Step environment (this publishes the commands)
        if IS_MOBILE:
            ts = env.step(action, base_action=recorder.current_base_action, get_base_vel=True)
            base_actions.append(recorder.current_base_action.copy())
        else:
            ts = env.step(action, get_base_vel=False)

        t2 = time.time()

        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])

        time.sleep(max(0, DT_RECORD - (time.time() - t0)))

    actual_timesteps = len(actions)

    # Start continuous background hold thread immediately to prevent any gap
    hold_left = list(recorder.current_left_joints)
    hold_right = list(recorder.current_right_joints)
    hold_left_grip = recorder._normalize_gripper(recorder.current_left_gripper)
    hold_right_grip = recorder._normalize_gripper(recorder.current_right_gripper)
    hold_stop = threading.Event()

    def _hold_loop():
        while not hold_stop.is_set():
            _hold_arms(env, hold_left, hold_right, hold_left_grip, hold_right_grip)
            time.sleep(DT)

    hold_thread = threading.Thread(target=_hold_loop, daemon=True)
    hold_thread.start()

    # Handle discard (Square button) — hold thread is now running
    if recorder.should_discard:
        hold_stop.set()
        hold_thread.join()
        return False, False

    print(f'\nRecorded {actual_timesteps} timesteps')
    print(f'Avg fps: {actual_timesteps / (time.time() - time0):.2f}')

    if actual_timesteps < 10:
        print('Too few timesteps recorded, discarding...')
        hold_stop.set()
        hold_thread.join()
        return False, False

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        print(f'\n\nfreq_mean is {freq_mean:.2f}, lower than 30, re-collecting... \n\n')
        hold_stop.set()
        hold_thread.join()
        return False, False

    # Check for None images
    for cam_name in camera_names:
        none_count = sum(
            1 for ts in timesteps[:actual_timesteps]
            if ts.observation['images'].get(cam_name) is None
        )
        if none_count > 0:
            print(f'\nERROR: {cam_name} has {none_count}/{actual_timesteps} None images!')
            print(f'Check if camera {cam_name} is connected and streaming.')
            hold_stop.set()
            hold_thread.join()
            return False, False

    # Save in LeRobot format (hold thread keeps running in background)
    save_error = [None]

    def _save():
        try:
            for i in range(actual_timesteps):
                ts = timesteps[i]
                frame = {
                    'observation.state': ts.observation['qpos'],
                    'observation.velocity': ts.observation['qvel'],
                    'observation.effort': ts.observation['effort'],
                    'action': actions[i],
                }
                if IS_MOBILE:
                    frame['action.base'] = base_actions[i]
                for cam_name in camera_names:
                    frame[f'observation.images.{cam_name}'] = (
                        ts.observation['images'][cam_name]
                    )
                lerobot_writer.add_frame(frame)
            lerobot_writer.save_episode(task=task_name or 'default')
        except Exception as e:
            save_error[0] = e

    t0 = time.time()
    save_thread = threading.Thread(target=_save, daemon=True)
    save_thread.start()
    save_thread.join()
    print(f'LeRobot saving: {time.time() - t0:.1f} secs')

    # Stop hold thread now that save is done (teleop loop will take over)
    hold_stop.set()
    hold_thread.join()

    if save_error[0] is not None:
        print(f'ERROR during save: {save_error[0]}')
        return False, False

    return True, False  # (success, exit_requested)


def _hold_arms(env, left_joints, right_joints, left_gripper=None, right_gripper=None):
    """Send position commands to both arms (and optionally grippers) without reading cameras."""
    env.follower_bot_left.arm.set_joint_positions(left_joints, blocking=False)
    env.follower_bot_right.arm.set_joint_positions(right_joints, blocking=False)
    if left_gripper is not None and right_gripper is not None:
        env.set_gripper_pose(left_gripper, right_gripper)


def signal_handler(sig, frame):
    print('\nYou pressed Ctrl+C!')
    exit(1)


def main(args: dict):
    signal.signal(signal.SIGINT, signal_handler)

    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    print(f'Dataset name: episode_*')

    node = create_interbotix_global_node('aloha')

    # Create environment with base torque enabled
    env = make_real_env(
        node=node,
        setup_robots=False,
        setup_base=IS_MOBILE,
        torque_base=True,  # Enable base motor torque for joystick control
    )

    robot_startup(node)

    # Initialize arms
    opening_ceremony(env.follower_bot_left, env.follower_bot_right)

    # Hold arms at start pose during remaining init (prevents sag)
    start_arm_qpos = START_ARM_POSE[:6]
    init_hold_stop = threading.Event()

    def _init_hold_loop():
        while not init_hold_stop.is_set():
            _hold_arms(env, list(start_arm_qpos), list(start_arm_qpos))
            time.sleep(DT)

    init_hold_thread = threading.Thread(target=_init_hold_loop, daemon=True)
    init_hold_thread.start()

    # Get base reference if mobile
    base = env.base if IS_MOBILE else None

    # Create Cartesian joystick recorder (teleop mode)
    recorder = CartesianRecorderTeleop(
        node,
        env.follower_bot_left,
        env.follower_bot_right,
        env,
        base
    )

    # Determine starting episode index (based on existing LeRobot parquet files)
    output_name = args.get('subtask') or 'subtask'
    if os.path.isabs(output_name):
        lerobot_dir = output_name
    else:
        lerobot_dir = os.path.join(dataset_dir, output_name)
    episode_idx = args['episode_idx'] if args['episode_idx'] is not None else get_auto_index(lerobot_dir)

    # LeRobot writer
    lerobot_features = build_aloha_features(camera_names, IS_MOBILE)
    lerobot_writer = LeRobotWriter.create(
        root=lerobot_dir,
        repo_id=f'aloha/{args["task_name"]}',
        fps=FPS,
        features=lerobot_features,
        robot_type='aloha_vx300s',
        start_episode=episode_idx,
    )
    print(f'LeRobot output -> {lerobot_dir} (starting at episode {episode_idx})')

    print('\nWaiting for PS4 controller...')

    # Stop init hold — teleop loop's recorder.update() takes over at 50Hz
    init_hold_stop.set()
    init_hold_thread.join()

    while rclpy.ok():
        success, exit_requested = capture_one_episode(
            recorder,
            env,
            max_timesteps,
            camera_names,
            lerobot_writer=lerobot_writer,
            task_name=args['task_description'],
        )

        if exit_requested:
            print('\n>>> Exit requested. Moving to working position...')
            break

        # Hold arms between episodes
        _hold_arms(env, list(recorder.current_left_joints), list(recorder.current_right_joints),
                   recorder._normalize_gripper(recorder.current_left_gripper),
                   recorder._normalize_gripper(recorder.current_right_gripper))

        if success:
            print(f'\n>>> Episode {episode_idx} saved successfully!')
            episode_idx += 1
        else:
            lerobot_writer.discard_episode()

        # Reset recorder state for next episode (arms stay in current position)
        recorder.reset_for_new_episode()
        _hold_arms(env, list(recorder.current_left_joints), list(recorder.current_right_joints),
                   recorder._normalize_gripper(recorder.current_left_gripper),
                   recorder._normalize_gripper(recorder.current_right_gripper))

        print(f'\nReady for next episode (will be episode {episode_idx})')
        print('Arms staying at current position. Use Options button to go home if needed.')

    # Hold arms continuously until move_arms takes over (prevents sag gap)
    exit_hold_stop = threading.Event()

    def _exit_hold_loop():
        while not exit_hold_stop.is_set():
            _hold_arms(env, list(recorder.current_left_joints), list(recorder.current_right_joints),
                       recorder._normalize_gripper(recorder.current_left_gripper),
                       recorder._normalize_gripper(recorder.current_right_gripper))
            time.sleep(DT)

    exit_hold_thread = threading.Thread(target=_exit_hold_loop, daemon=True)
    exit_hold_thread.start()

    # Before exiting, move arms to home (work) pose
    print('\n>>> Moving arms to work pose before exit...')
    exit_hold_stop.set()
    exit_hold_thread.join()
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [env.follower_bot_left, env.follower_bot_right],
        [start_arm_qpos, start_arm_qpos],
        moving_time=4.0,
    )

    # Finalize LeRobot dataset (writes metadata files)
    lerobot_writer.finalize()

    print('\n>>> Recording session complete!')
    robot_shutdown()


def get_auto_index(lerobot_dir):
    """Return the next episode index by counting existing parquet files in data/chunk-*."""
    data_dir = os.path.join(lerobot_dir, 'data')
    if not os.path.isdir(data_dir):
        return 0
    count = 0
    for chunk in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk)
        if os.path.isdir(chunk_path):
            count += len([f for f in os.listdir(chunk_path) if f.endswith('.parquet')])
    return count


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    freq_mean = 1 / dt_mean
    print((
        f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} '
        f'Step env: {np.mean(step_env_time):.3f}')
    )
    return freq_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Record ALOHA episodes with Cartesian PS4 joystick control (Teleop Mode)'
    )
    parser.add_argument(
        '--task_name',
        action='store',
        type=str,
        help='Task name.',
        required=True,
    )
    parser.add_argument(
        '--episode_idx',
        action='store',
        type=int,
        help='Starting episode index.',
        default=None,
        required=False,
    )
    parser.add_argument(
        '--subtask',
        action='store',
        type=str,
        help='Subtask name, used as output folder and default task description.',
        default='subtask',
    )
    parser.add_argument(
        '--task_description',
        action='store',
        type=str,
        help='Task description for LeRobot metadata (default: same as --subtask).',
        default=None,
    )
    args = vars(parser.parse_args())
    if args['task_description'] is None:
        args['task_description'] = args['subtask']
    main(args)
