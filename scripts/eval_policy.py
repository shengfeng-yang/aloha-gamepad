#!/usr/bin/env python3
"""Evaluate a trained pi0.5 policy on the real ALOHA robot.

Loads a pretrained pi0.5 checkpoint and runs it on the real robot,
replacing joystick teleop with learned neural network actions.

The robot arms move to home position first, then the policy takes over.
Press Ctrl+C to stop at any time.

Usage:
    python eval_pi05_real.py --checkpoint /path/to/pretrained_model
    python eval_pi05_real.py --checkpoint /path/to/pretrained_model --max_timesteps 750
    python eval_pi05_real.py --checkpoint /path/to/pretrained_model --task_name aloha_mobile_tube_transfer
"""

import argparse
import signal
import sys
import time

import numpy as np
import torch

# ROS2 / ALOHA imports
from aloha.constants import (
    DT,
    FPS,
    IS_MOBILE,
    START_ARM_POSE,
    TASK_CONFIGS,
)
from aloha.real_env import make_real_env
from aloha.robot_utils import (
    move_arms,
    move_grippers,
    setup_follower_bot,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

# LeRobot imports
sys.path.insert(0, "/home/aloha/lerobot/src")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


TASK_STRING = "Transfer the tube from one rack to another rack"
CAMERA_NAMES = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


def signal_handler(sig, frame):
    print('\nCtrl+C pressed - stopping...')
    sys.exit(0)


def load_policy(checkpoint_path, device):
    """Load pi0.5 policy and preprocessor/postprocessor."""
    print(f"Loading pi0.5 from {checkpoint_path}...")
    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Model loaded: {total_params / 1e6:.0f}M parameters")

    print("Loading preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )
    print("  Loaded.")

    return policy, preprocessor, postprocessor


def get_observation_dict(env_obs):
    """Convert RealEnv observation to the dict format expected by the policy.

    RealEnv returns:
        obs['qpos']:   14D (6 left arm + 1 left gripper + 6 right arm + 1 right gripper)
        obs['qvel']:   14D (same layout)
        obs['effort']: 14D (7 left + 7 right, no gripper split)
        obs['images']: dict of (480, 640, 3) uint8

    Policy expects:
        observation.state:  14D float32
        observation.velocity: 14D float32
        observation.effort: 14D float32
        observation.images.cam_*: (480, 640, 3) uint8
    """
    obs = {
        "observation.state": env_obs['qpos'].astype(np.float32),
        "observation.velocity": env_obs['qvel'].astype(np.float32),
        "observation.effort": env_obs['effort'].astype(np.float32),
    }
    for cam_name in CAMERA_NAMES:
        img = env_obs['images'].get(cam_name)
        if img is not None:
            obs[f"observation.images.{cam_name}"] = img.astype(np.uint8)
        else:
            obs[f"observation.images.{cam_name}"] = np.zeros(
                (480, 640, 3), dtype=np.uint8
            )
    return obs


def run_policy_episode(
    env,
    policy,
    preprocessor,
    postprocessor,
    device,
    max_timesteps=1500,
    task_string=TASK_STRING,
):
    """Run one episode with the trained policy on the real robot."""
    # Reset preprocessor/postprocessor state
    preprocessor.reset()
    postprocessor.reset()
    policy._action_queue.clear()

    # Get initial observation
    ts = env.reset(fake=True)  # fake=True: don't reset joints, just get obs
    dt_target = 1.0 / FPS

    print(f"\nRunning policy for up to {max_timesteps} steps ({max_timesteps / FPS:.1f}s)...")
    print("Press Ctrl+C to stop.\n")

    # Countdown before starting
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    print("  GO!\n")

    for step in range(max_timesteps):
        t0 = time.time()

        # Get observation from robot
        env_obs = env.get_observation(get_base_vel=False)
        obs_dict = get_observation_dict(env_obs)

        # Run policy inference
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" else torch.inference_mode(),
        ):
            obs_tensor = prepare_observation_for_inference(
                obs_dict, device, task=task_string, robot_type="aloha_vx300s"
            )
            obs_tensor = preprocessor(obs_tensor)
            action_tensor = policy.select_action(obs_tensor)
            action_tensor = postprocessor(action_tensor)

        # Convert action to numpy
        # action shape: [1, action_dim] -> [action_dim]
        action_np = action_tensor.squeeze(0).cpu().numpy()

        # Split arm action (14D) and base action (2D) if present
        arm_action = action_np[:14]
        base_action = None
        if IS_MOBILE and len(action_np) > 14:
            base_action = action_np[14:16]

        # Send to robot
        env.step(arm_action, base_action=base_action, get_obs=False)

        # Timing
        elapsed = time.time() - t0
        sleep_time = max(0, dt_target - elapsed)
        time.sleep(sleep_time)

        # Status print
        if step % 50 == 0:
            fps_actual = 1.0 / max(elapsed, 1e-6)
            print(f"  Step {step}/{max_timesteps}  "
                  f"inference: {elapsed*1000:.0f}ms  "
                  f"fps: {fps_actual:.1f}")

    print(f"\nEpisode done ({max_timesteps} steps).")


def main():
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Evaluate pi0.5 on real ALOHA robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/aloha/aloha_data/checkpoints/020000/pretrained_model",
        help="Path to pretrained_model directory",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="aloha_mobile_tube_transfer",
        help="Task name from TASK_CONFIGS (for episode_len and camera_names)",
    )
    parser.add_argument(
        "--task_string",
        type=str,
        default=TASK_STRING,
        help="Task description string for the policy",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=None,
        help="Max timesteps per episode (default: from TASK_CONFIGS or 1500)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--no_reset",
        action="store_true",
        help="Don't reset arms to home before first episode",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Max timesteps
    max_timesteps = args.max_timesteps
    if max_timesteps is None:
        if args.task_name in TASK_CONFIGS:
            max_timesteps = TASK_CONFIGS[args.task_name]['episode_len']
        else:
            max_timesteps = 1500
    print(f"Max timesteps: {max_timesteps} ({max_timesteps / FPS:.1f}s)")

    # Load policy
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device)

    # Initialize robot
    print("\nInitializing robot...")
    node = create_interbotix_global_node('aloha')
    env = make_real_env(
        node=node,
        setup_robots=True,
        setup_base=IS_MOBILE,
        torque_base=IS_MOBILE,
    )
    robot_startup(node)

    if not args.no_reset:
        print("Moving arms to home position...")
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            [env.follower_bot_left, env.follower_bot_right],
            [start_arm_qpos, start_arm_qpos],
            moving_time=4.0,
        )

    # Run episodes
    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")

        if ep > 0:
            input("Press Enter to start next episode (or Ctrl+C to exit)...")

        run_policy_episode(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            max_timesteps=max_timesteps,
            task_string=args.task_string,
        )

    # Cleanup
    print("\nMoving arms to home position...")
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [env.follower_bot_left, env.follower_bot_right],
        [start_arm_qpos, start_arm_qpos],
        moving_time=4.0,
    )

    print("Done!")
    robot_shutdown()


if __name__ == "__main__":
    main()
