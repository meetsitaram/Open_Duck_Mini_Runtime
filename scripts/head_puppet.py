"""
Sets up the robot in init position, you control the head with the xbox controller
"""

import time
import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.xbox_controller import XBoxController


from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector

duck_config = DuckConfig()

xbox_controller = XBoxController(50, only_head_control=True)

# on raspberry pi set higher volume using this command
#  amixer cset numid=3 30%

if duck_config.speaker:
    sounds = Sounds(volume=100.0, sound_directory="../mini_bdx_runtime/assets/")
if duck_config.antennas:
    antennas = Antennas()
if duck_config.eyes:
    eyes = Eyes()
if duck_config.projector:
    projector = Projector()

hwi = HWI(duck_config)

hwi.override_init_pos({
    "left_hip_yaw": 0.0398835,
    "left_hip_roll": 0.02454369,
    "left_hip_pitch": -0.84062147,
    "left_knee": 1.81930122,
    "left_ankle": -0.93112633,
    "neck_pitch": 1.116,
    "head_pitch": -1.155,
    "head_yaw": -0.0,
    "head_roll": 0.008,
    # "left_antenna": 0,
    # "right_antenna": 0,
    "right_hip_yaw": 0.02607767,
    "right_hip_roll": -0.02300971,
    "right_hip_pitch": 1.01089333,
    "right_knee": 1.78555364,
    "right_ankle": -0.76238845,
})

kps = [8] * 14
kps[7] = 4 # slow down the head roll movement - not so good motor 
            # (roll and yaw motors got mixed up)
# kps = [8] * 14
kds = [0] * 14

hwi.set_kps(kps)
hwi.set_kds(kds)
hwi.turn_on()

limits = {
    "neck_pitch": [-20, 60],
    "head_pitch": [-60, 15],
    "head_yaw": [-50, 50],  # (roll and yaw motors got mixed up)
    "head_roll": [-20, 20],  # (roll and yaw motors got mixed up)
}

try:
    while True:

        last_commands, buttons, left_trigger, right_trigger = (
            xbox_controller.get_last_command()
        )

        l_x = last_commands[5]
        l_y = last_commands[4]
        r_x = last_commands[6]
        r_y = last_commands[3]

        print('last_commands:', last_commands)

        head_yaw_deg = (
            l_x * (limits["head_yaw"][1] - limits["head_yaw"][0]) / 2
            + (limits["head_yaw"][1] + limits["head_yaw"][0]) / 2
        )
        # print("Head Yaw (deg): ", head_yaw_deg)
        head_yaw_pos_rad = np.deg2rad(head_yaw_deg)
        # print("Head Yaw (deg): ", head_yaw_deg)

        head_roll_deg = (
            r_x * (limits["head_roll"][1] - limits["head_roll"][0]) / 2
            + (limits["head_roll"][1] + limits["head_roll"][0]) / 2
        )
        head_roll_deg = head_roll_deg - 15 # to keep it straight
        print("Head Roll (deg): ", head_roll_deg)
        head_roll_pos_rad = np.deg2rad(head_roll_deg)

        head_pitch_deg = (
            l_y * (limits["head_pitch"][1] - limits["head_pitch"][0]) / 2
            + (limits["head_pitch"][1] + limits["head_pitch"][0]) / 2
        )
        head_pitch_pos_rad = np.deg2rad(head_pitch_deg)

        neck_pitch_deg = (
            r_y * (limits["neck_pitch"][1] - limits["neck_pitch"][0]) / 2
            + (limits["neck_pitch"][1] + limits["neck_pitch"][0]) / 2
        )
        neck_pitch_pos_rad = np.deg2rad(neck_pitch_deg)

        # print("head position (deg): ", head_yaw_deg, head_roll_deg, head_pitch_deg)
        # print("Head positions (rad): ", head_yaw_pos_rad, head_roll_pos_rad, head_pitch_pos_rad)

        hwi.set_position("head_yaw", head_yaw_pos_rad)
        hwi.set_position("head_roll", head_roll_pos_rad)
        hwi.set_position("head_pitch", head_pitch_pos_rad)
        hwi.set_position("neck_pitch", neck_pitch_pos_rad)

        if duck_config.antennas:
            antennas.set_position_left(right_trigger)
            antennas.set_position_right(left_trigger)

        if buttons.B.triggered:
            if duck_config.speaker:
                sounds.play_random_sound()

        if buttons.X.triggered:
            if duck_config.projector:
                projector.switch()

        # pygame.event.pump()  # process event queue
        time.sleep(1 / 30)
except KeyboardInterrupt:
    if duck_config.antennas:
        antennas.stop()
