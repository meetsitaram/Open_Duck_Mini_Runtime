import time
import pickle
import json

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI

from mini_bdx_runtime.rl_utils import make_action_dict
from mini_bdx_runtime.duck_config import DuckConfig

import os

HOME_DIR = os.path.expanduser("~")


class RecordAndReplay:

    def __init__(
        self,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        commands=False,
        pitch_bias=0,
        save_obs=False,
        save_as=None,
        replay_obs=None,
        display=False,
        machine='pc',
    ):

        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.machine = machine
        self.commands = commands
        if self.machine == 'pc':
            self.commands = False
        self.pitch_bias = pitch_bias
        self.display = display
        self.save_as = save_as

        self.num_dofs = 14

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []
            self.saved_positions = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.hwi = HWI(self.duck_config, serial_port)

        self.init_pos = list(self.hwi.init_pos.values())
        print("Init pos:", self.init_pos)

        self.motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz

    def override_init_pos(self):
                
        self.hwi.override_init_pos({
            "left_hip_yaw": -0.174,
            "left_hip_roll": -0.123,
            "left_hip_pitch": -0.875 ,
            "left_knee": 1.848 ,
            "left_ankle": -0.678,
            
            "neck_pitch": 1.116,
            "head_pitch": -1.155,
            "head_yaw": -0.146,
            "head_roll": 0.008,
            # "left_antenna": 0,
            # "right_antenna": 0,
            "right_hip_yaw": -0.078,
            "right_hip_roll": 0.203 ,
            "right_hip_pitch": 0.993 ,
            "right_knee": 1.848,
            "right_ankle": -0.677,
        })
                
    def get_obs(self):


        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad


        if dof_pos is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        obs = np.concatenate(
            [
                dof_pos
            ]
        )

        return obs

    def print_obs(self, obs):
        if obs is None:
            return 
        # print("Observations:")
        # print(f"  Gyro: {obs[0:3]}")
        # print(f"  Accelero: {obs[3:6]}")
        # print(f"  Commands: {obs[6:13]}")
        # print(f"  DOF Pos (rel): {obs[13:27]}")
        # print(f"  DOF Vel: {obs[27:41]}")
        # print(f"  Feet Contacts: {obs[41:45]}")
        # print(f"  Motor Targets: {obs[45:59]}")
        print(f"left leg: {obs[0:4]} right leg: {obs[8:13]} head: {obs[4:8]}")


    def run(self):
        try:
            print("Starting")
            piano_robot_pos = {
                'left': {
                    'up': {},
                    'down': {}
                },
                'right': {
                    'up': {},
                    'down': {}
                }
            }
            if os.path.exists("piano_positions.json"):
                with open("piano_positions.json", "r") as f:
                    piano_robot_pos = json.load(f)
                print("Loaded existing piano_positions.json")

            for leg in ['left', 'right']:
                for note in ['0', 'e', 'f', 'g', 'a', 'b', 'c', 'd']:
                    for action in ['up', 'down']:

                        print(f"Move {leg} leg {action} at {note}. Press Enter to capture, 's' to skip.")
                        user_input = input().strip().lower()
                        if user_input == 's':
                            print(f"Skipped capturing for {leg} leg {action} at {note}")
                            continue
                        obs = self.get_obs()
                        if obs is None:
                            print("Failed to get observations. Try again.")
                            obs = self.get_obs()

                            
                        print("current obs:", obs)
                        if leg == 'left':
                            obs = obs[0:5]
                        else:
                            obs = obs[9:14]

                        print(f"Captured position for {leg} leg {action} at {note}: {obs}")

                        piano_robot_pos[leg][action][note] = obs.tolist()
                        with open("piano_positions.json", "w") as f:
                            json.dump(piano_robot_pos, f, indent=4)

            print("Piano robot positions captured:", piano_robot_pos)
            # save the positions to paino_positions.json
            with open("piano_positions.json", "w") as f:
                json.dump(piano_robot_pos, f, indent=4)
            print("Positions saved to piano_positions.json")

        except KeyboardInterrupt:
            pass

        print("DONE capturing piano robot positions.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        action="store_true",
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        required=False,
        default=None,
        help="save the run's observations to this file",
    )    
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="display observations in the console",
    )
    parser.add_argument('--machine', choices=['pc', 'pi'], default='pc')

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RecordAndReplay(
        duck_config_path=args.duck_config_path,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        save_as=args.save_as,
        replay_obs=args.replay_obs,
        display=args.display,
        machine=args.machine,
    )
    print("Done with RecordAndReplay")
    rl_walk.run()
