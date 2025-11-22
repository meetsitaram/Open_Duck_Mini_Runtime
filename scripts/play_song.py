import time
import pickle
import json

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI

from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
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
        piano_positions=None,
        song_file=None,
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

        self.piano_positions = piano_positions
        self.song_file = song_file
        if self.piano_positions is not None and self.song_file is not None:
            with open(self.piano_positions, 'r') as f:
                self.positions = json.load(f)
            with open(self.song_file, 'r') as f:
                song_notes = json.load(f)
            self.note_positions = []
            for step in song_notes:
                left_note = step['left']['note']
                left_position = step['left']['position']
                right_note = step['right']['note']
                right_position = step['right']['position']
                duration = step['duration']
                left_pos = self.positions['left'][left_position][left_note] if left_note != '0' else self.positions['left']['up']['e']
                right_pos = self.positions['right'][right_position][right_note] if right_note != '0' else self.positions['right']['up']['e']
                
                head_pos = None
                if 'head_move' in step.keys():
                    head_pos = self.positions['head'][step['head_move']] 

                sound = None
                if 'sound' in step.keys():
                    sound = step['sound']

                self.note_positions.append((left_pos, right_pos, duration, head_pos, sound))


        self.hwi = HWI(self.duck_config, serial_port)


        # do not start if we are only saving observations
        self.override_init_pos()
        if not self.save_obs:
            # self.override_init_pos()
            self.start()

        self.imu = None

        self.feet_contacts = None

        self.init_pos = list(self.hwi.init_pos.values())
        print("Init pos:", self.init_pos)

        self.motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz
        self.xbox_controller = None
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        # Optional expression features
        self.eyes = None
        self.projector = None
        self.sounds = None
        self.antennas = None
        if self.machine != 'pc':
            if self.duck_config.eyes:
                self.eyes = Eyes()
            if self.duck_config.projector:
                self.projector = Projector()
            if self.duck_config.speaker:
                self.sounds = Sounds(
                    volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
                )
            if self.duck_config.antennas:
                self.antennas = Antennas()

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

        imu_data = {'gyro': np.zeros(3), 'accelero': np.zeros(3)} # dummy data

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        cmds = self.last_commands

        feet_contacts = np.zeros(4) # dummy data

        obs = np.concatenate(
            [
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos, # - self.init_pos,
                dof_vel * 0.05,
                feet_contacts,
                self.motor_targets,
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
        print(f"left leg positions: {obs[13:18]} right leg positions: {obs[22:27]}")

    def start(self):
        # low_kps_pid = [2, 0, 0]
        # kps = [low_kps_pid[0]] * 14 # 14
        kds = [self.pid[2]] * 14 # 14
        kps = [self.pid[0]] * 14 # 14
        # kds = [self.pid[2]] * 14 # 14

        # lower head kps
        kps[5:9] = [6, 6, 6, 6]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def reset_and_stop(self):
        self.hwi.turn_on()
        time.sleep(2)       
        self.hwi.turn_off()


    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    # if self.buttons.dpad_right.triggered:
                    #     obs = self.get_obs()
                    #     if obs is None:
                    #         continue
                    #     print("obs:", obs)

                    if self.buttons.X.triggered:
                        if self.projector is not None:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        print("B triggered for speaker sound")
                        if self.sounds is not None:
                            self.sounds.play_random_sound()

                    if self.antennas is not None:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                if self.paused:
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue

                if self.display:
                    self.print_obs(obs)

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.piano_positions is not None:
                    if i < len(self.note_positions):
                        left_pos, right_pos, duration, head_pos, sound = self.note_positions[i]
                        current_pos = self.hwi.get_present_positions(
                            ignore=[
                                "left_antenna",
                                "right_antenna",
                            ]
                        )
                        print("current position:", current_pos, "note i:", i)
                        if current_pos is None:
                            print("failed to read current positions, skipping")
                            continue
                        current_pos[0:5] = left_pos
                        if head_pos is not None:
                            current_pos[5:9] = head_pos
                        current_pos[9:14] = right_pos

                        self.motor_targets = np.array(current_pos)

                        action_dict = make_action_dict(
                            self.motor_targets, list(self.hwi.joints.keys())
                        )
                        self.hwi.set_position_all(action_dict)
                        print(f"Setting positions: left {left_pos}, right {right_pos}, duration {duration}")
                        
                        if sound is not None and self.sounds is not None:
                            self.sounds.play(sound)

                        time.sleep(duration)
                    else:
                        print("BREAKING ")
                        break
                # else:
                    # In normal mode, update motor_targets for observation, but don't set positions
                    # self.motor_targets = obs[13:27] + np.array(self.init_pos)
                    # self.motor_targets = np.array(self.init_pos.copy())

                i += 1

                took = time.time() - t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

                # print("actual obs:")
                # self.print_obs(self.get_obs())

        except KeyboardInterrupt:
            if self.antennas is not None:
                self.antennas.stop()
            if self.eyes is not None:
                self.eyes.stop()
            if self.projector is not None:
                self.projector.stop()
            if self.feet_contacts is not None:
                self.feet_contacts.stop()

        if self.save_obs:
            save_file = self.save_as if self.save_as is not None else "robot_saved_obs.pkl"
            pickle.dump(self.saved_obs, open(save_file, "wb"))
        print("TURNING OFF")

        self.reset_and_stop()


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
        "--piano_positions",
        type=str,
        required=False,
        default="piano_positions.json",
        help="path to piano positions JSON file",
    )
    parser.add_argument(
        "--song_file",
        type=str,
        required=False,
        default="song.json",
        help="path to song JSON file",
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
        piano_positions=args.piano_positions,
        song_file=args.song_file,
        display=args.display,
        machine=args.machine,
    )
    print("Done with RecordAndReplay")
    rl_walk.run()
