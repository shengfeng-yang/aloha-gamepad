# ALOHA Gamepad

#### Project Website: https://shengfeng-yang.github.io/aloha-gamepad/

Untethered teleoperation for [Mobile ALOHA](https://mobile-aloha.github.io/) using a gamepad + inverse kinematics. Control both bimanual arms and mobile base with a standard wireless controller (PS4). Includes data collection, [pi0.5](https://www.physicalintelligence.company/blog/pi05) fine-tuning, and inference pipelines.

## Overview

This project replaces ALOHA's leader-arm teleoperation with a PS4 gamepad, enabling untethered control of the bimanual arms and mobile base through Cartesian-space commands and real-time IK. Collected demonstrations are saved in [LeRobot](https://github.com/huggingface/lerobot) v3.0 format for direct use with pi0.5 fine-tuning.

## Repository Structure

```
aloha-gamepad/
├── launch/
│   └── aloha_bringup.launch.py   # ROS2 launch file for ALOHA hardware
├── scripts/
│   ├── gamepad_teleop.py          # Gamepad teleoperation + data recording
│   ├── lerobot_writer.py          # LeRobot v3.0 dataset writer
│   ├── eval_policy.py             # Run fine-tuned pi0.5 on the real robot
│   ├── run_eval.sh                # Shell wrapper for policy evaluation
│   ├── sleep_arms.py              # Send arms to sleep pose
│   └── view_dataset.py            # Visualize collected datasets
```

## Workflow

### 1. Launch ALOHA Hardware

Start the ROS2 nodes for the follower arms, cameras, and mobile base:

```bash
ros2 launch aloha aloha_bringup.launch.py
```

Or use the custom launch file in this repo if you've modified the configuration:

```bash
ros2 launch launch/aloha_bringup.launch.py
```

### 2. Teleoperate and Record Episodes

Run the gamepad teleoperation script with a PS4 controller connected:

```bash
python scripts/gamepad_teleop.py --task_name <task_name>
```

**Gamepad controls:**

| Control | Action |
|---|---|
| Left Stick | Move forward/backward (Y), left/right (X) |
| Right Stick | Move up/down (Y), roll (X) |
| R2 + Left Stick | Pitch / Yaw orientation control |
| L2 + Right Stick | Base movement (forward/back + rotation) |
| Triangle / X | Open / Close gripper |
| L1 / R1 | Select left / right arm (L1+R1 for both) |
| D-Pad Up/Down | Speed adjustment |
| Circle | Start / Stop recording |
| Square | Discard current recording |
| Options / Share | Home pose (single arm / both arms) |
| PS Button | Exit program |

Episodes are saved in LeRobot v3.0 format using `lerobot_writer.py`.

### 3. Sleep Arms

After finishing data collection, send the arms to their sleep pose:

```bash
python scripts/sleep_arms.py
```

### 4. Visualize Collected Data

View dataset summary and plot individual episodes:

```bash
# Print dataset summary
python scripts/view_dataset.py /path/to/dataset --summary

# Plot a specific episode (all cameras)
python scripts/view_dataset.py /path/to/dataset --episode 0

# Plot a specific camera
python scripts/view_dataset.py /path/to/dataset --episode 0 --camera cam_high
```

### 5. Fine-tune pi0.5

Fine-tune pi0.5 on your collected dataset using the [LeRobot training pipeline](https://github.com/huggingface/lerobot). The collected data is already in the compatible v3.0 format.

### 6. Evaluate Fine-tuned Policy

Run the fine-tuned pi0.5 model on the real robot:

```bash
# Using the shell wrapper (sets up ROS2 + conda environment)
./scripts/run_eval.sh --checkpoint /path/to/pretrained_model

# Or directly
python scripts/eval_policy.py --checkpoint /path/to/pretrained_model --episodes 3
```

## Dependencies

- ROS2 Humble
- [Interbotix ROS packages](https://github.com/Interbotix/interbotix_ros_manipulators) (aloha, interbotix_xs_modules, etc.)
- Python 3.10+
- numpy, torch, pyarrow, imageio[ffmpeg]
- [LeRobot](https://github.com/huggingface/lerobot) (for pi0.5 fine-tuning and evaluation)
- PS4 controller (connected via Bluetooth or USB)

## Acknowledgments

Built on top of [Mobile ALOHA](https://mobile-aloha.github.io/) by Tony Zhao et al., [ALOHA](https://github.com/Interbotix/aloha) by Interbotix, and [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face.
