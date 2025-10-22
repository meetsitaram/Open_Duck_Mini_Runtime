conda create -y -n duckdog python=3.10
conda activate duckdog


### configure motors
    "left_hip_yaw": 20,
    "left_hip_roll": 21,
    "left_hip_pitch": 22,
    "left_knee": 23,
    "left_ankle": 24,

    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,

    "neck_pitch": 30,
    "head_pitch": 31,
    "head_yaw": 32,
    "head_roll": 33,

### calibrate using lerobot utility
lerobot-calibrate --robot.type=openduckmini_follower --robot.port=COM5 --robot.id=openduckmini_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\Open_Duck_Mini_Runtime"
- turns out this actually updates all the min and max position limits into the servo memory

- before doing the lerobot calibrate, find_soft_offsets.py was causing the motors go into strange positions. but once that calibration is done, find_soft_offsets.py seems to keep the robot in nice position, while trying to identity the motor offsets. i also made some minor chagnes to this script to not reset all motor positions in one shot, and instead do one at a time.

- changed mini_bdx_runtime/mini_bdx_runtime/rustypot_position_hwi.py, scripts/check_voltage.py to use COM5 port

- while having both lerobot and openduckmini-runtime installed in the same venv, there were some conflicts related to numpy version. changed setup.cfg to have onnxruntime==1.23.1 and numpy==2.2.6 to resolve the conflicts.
