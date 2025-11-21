import time
import pickle
import json

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI

# from mini_bdx_runtime.raw_imu import Imu
# from mini_bdx_runtime.xbox_controller import XBoxController
# from mini_bdx_runtime.feet_contacts import FeetContacts
# from mini_bdx_runtime.eyes import Eyes
# from mini_bdx_runtime.sounds import Sounds
# from mini_bdx_runtime.antennas import Antennas
# from mini_bdx_runtime.projector import Projector
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

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            with open(self.replay_obs, 'r') as f:
                self.positions = json.load(f)
            self.note_positions = []
            notes = ['e', 'f', 'g', 'a', 'b', 'c', 'd']
            is_first_left_up = True
            is_first_right_up = True
            for note in notes:
                # down position
                left_down = self.positions['left']['down'][note]
                
                # up position
                left_up = self.positions['left']['up'][note]

                if is_first_left_up:
                    self.note_positions.insert(0, ('left', 'up', left_up))
                    is_first_left_up = False

                print("appending note:", note, "left up pos:", left_up)
                self.note_positions.append(('left', 'up', left_up))
                self.note_positions.append(('left', 'down', left_down))
                self.note_positions.append(('left', 'up', left_up))

            for note in notes:
                # down position
                right_down = self.positions['right']['down'][note]
                
                # up position
                right_up = self.positions['right']['up'][note]

                if is_first_right_up:
                    self.note_positions.insert(0, ('right', 'up', right_up))
                    is_first_right_up = False
                    
                print("appending note:", note, "right up pos:", left_up)
                self.note_positions.append(('right', 'up', right_up))
                self.note_positions.append(('right', 'down', right_down))
                self.note_positions.append(('right', 'up', right_up))


        self.hwi = HWI(self.duck_config, serial_port)


        # do not start if we are only saving observations
        self.override_init_pos()
        if not self.save_obs:
            # self.override_init_pos()
            self.start()

        print("Initializing IMU")
        if self.machine != 'pc':
            self.imu = Imu(
                sampling_freq=int(self.control_freq),
                user_pitch_bias=self.pitch_bias,
                upside_down=self.duck_config.imu_upside_down,
            )
        else:
            self.imu = None

        self.feet_contacts = None
        if self.machine != 'pc':
            self.feet_contacts = FeetContacts()

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

        if self.imu is not None:
            imu_data = self.imu.get_data()
        else:
            imu_data = {'gyro': np.zeros(3), 'accelero': np.zeros(3)}

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

        if self.feet_contacts is not None:
            feet_contacts = self.feet_contacts.get()
        else:
            feet_contacts = np.zeros(4)

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
        # kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

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
                    if self.buttons.dpad_right.triggered:
                        obs = self.get_obs()
                        if obs is None:
                            continue
                        print("obs:", obs)

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

                if self.replay_obs is not None:
                    if i < len(self.note_positions):
                        (leg, up_down, pos) = self.note_positions[i]
                        current_pos = self.hwi.get_present_positions(
                            ignore=[
                                "left_antenna",
                                "right_antenna",
                            ]
                        )
                        print("current position:", current_pos, "note i:", i)
                        print("leg position to set:", pos, "note i:", i)
                        if current_pos is None:
                            print("failed to read current positions, skipping")
                            continue
                        if leg == 'left':
                            current_pos[0:5] = pos
                        else:
                            current_pos[9:14] = pos

                        if current_pos is not None:
                            self.motor_targets = np.array(current_pos)

                            action_dict = make_action_dict(
                                self.motor_targets, list(self.hwi.joints.keys())
                            )
                            self.hwi.set_position_all(action_dict)
                            print(f"Setting {leg} leg to position: {self.motor_targets}")
                            
                            if up_down == 'up':
                                time.sleep(0.2)
                            else:
                                time.sleep(1)
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
