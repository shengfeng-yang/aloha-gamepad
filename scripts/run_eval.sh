#!/bin/bash
# Launch pi0.5 evaluation on the real ALOHA robot.
# Usage:
#   ./scripts/run_eval.sh
#   ./scripts/run_eval.sh --max_timesteps 750
#   ./scripts/run_eval.sh --checkpoint /path/to/pretrained_model --episodes 3

set -e

# Source ROS2 and interbotix workspace
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash

# Activate conda environment with LeRobot/pi0.5 dependencies
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pi05_real

# Add ROS2 and interbotix packages to PYTHONPATH
export PYTHONPATH="\
/home/aloha/interbotix_ws/install/aloha/lib/python3.10/site-packages:\
/home/aloha/interbotix_ws/install/interbotix_xs_modules/lib/python3.10/site-packages:\
/home/aloha/interbotix_ws/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:\
/home/aloha/interbotix_ws/install/interbotix_slate_msgs/local/lib/python3.10/dist-packages:\
/home/aloha/interbotix_ws/install/interbotix_moveit_interface_msgs/local/lib/python3.10/dist-packages:\
/home/aloha/interbotix_ws/install/interbotix_common_modules/lib/python3.10/site-packages:\
/opt/ros/humble/lib/python3.10/site-packages:\
/opt/ros/humble/local/lib/python3.10/dist-packages:\
$PYTHONPATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/eval_policy.py" "$@"
