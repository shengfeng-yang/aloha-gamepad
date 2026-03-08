#!/usr/bin/env python3
"""
Sleep Arms Script

Sends follower arms to their sleep poses.
"""

from aloha.robot_utils import (
    sleep_arms,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def main():
    print('Sending follower arms to sleep pose...')

    node = create_interbotix_global_node('aloha')

    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )

    robot_startup(node)

    follower_bots = [follower_bot_left, follower_bot_right]

    for bot in follower_bots:
        torque_on(bot)

    sleep_arms(follower_bots, home_first=True)

    print('Done!')
    robot_shutdown(node)


if __name__ == '__main__':
    main()
