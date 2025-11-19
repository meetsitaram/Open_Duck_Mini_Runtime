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

- copied all new offests to duck_config.json


### captured offsets iin duck_config.json
- changed head_pitch to -0.150

    "joints_offsets": {
        "left_hip_yaw": 0.152,
        "left_hip_roll": 0.105,
        "left_hip_pitch": 0.856,
        "left_knee": 0.209,
        "left_ankle": -0.155,
        "neck_pitch": 0.065000000000000006,
        "head_pitch": -0.150,
        "head_yaw": -0.039999999999999994,
        "head_roll": 0.003,
        "right_hip_yaw": 0.029,
        "right_hip_roll": -0.007,
        "right_hip_pitch": -0.976,
        "right_knee": 0.134,
        "right_ankle": 0.165
    }


### action dict in code
tmp_action_dict: {'left_hip_yaw': np.float64(-0.0340741913318634), 'left_hip_roll': np.float64(0.10542618918418883), 'left_hip_pitch': np.float64(-0.5334340754151344), 'left_knee': np.float64(1.3523554440885783), 'left_ankle': np.float64(-0.6427498515844345), 'neck_pitch': np.float64(0.1498078554868698), 'head_pitch': np.float64(0.10449058562517166), 'head_yaw': np.float64(-0.021760093048214912), 'head_roll': np.float64(-0.14097267389297485), 'right_hip_yaw': np.float64(0.040854054063558576), 'right_hip_roll': np.float64(-0.15652061492204666), 'right_hip_pitch': np.float64(0.8372294402122498), 'right_knee': np.float64(1.457813336789608), 'right_ankle': np.float64(-0.6618384382724762)}



### piano 
-saved
  Motor Targets: [-0.143 -0.037 -0.406  1.015 -0.641 -0.503  0.299 -0.024  0.025 -0.015
  0.05   0.448  0.964 -0.562]

  DOF Pos (rel): [-0.145 -0.09   0.224 -0.353  0.143 -0.503  0.299 -0.024  0.025 -0.012
  0.115 -0.187 -0.415  0.234]


-replayed
  Motor Targets: [-0.152 -0.159 -0.398  1.514 -1.033 -0.506  0.159 -0.015  0.057  0.
  0.127  0.662  1.443 -0.682]
  
    DOF Pos (rel): [-0.15  -0.072  0.221 -0.362  0.248 -0.498  0.264 -0.027  0.028 -0.007
  0.093 -0.195 -0.424  0.236]

