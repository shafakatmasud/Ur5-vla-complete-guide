# UR5 VLA complete guide: connect, teleoperate, record, convert, train and run

This is the operating guide for our **physical UR5**, Logitech joystick, two RealSense cameras and Robotiq electric gripper. The commands below use the lab's existing addresses, workspaces, camera serial numbers, scripts and model directories. Reviewed against the source on **2026-11-05**.

Follow these five chapters in order. Start each long-running ROS process in its own terminal. For another recording session, reuse the running driver/cameras instead of launching duplicates.

1. [Connect the UR5 to the remote PC](#1-connect-the-ur5-to-the-remote-pc)
2. [Connect and use teleop](#2-connect-and-use-teleop)
3. [Prepare and collect a dataset](#3-prepare-and-collect-a-dataset)
4. [Zip the recordings and convert to LeRobot](#4-zip-the-recordings-and-convert-to-lerobot)
5. [Train and run each model](#5-train-and-run-each-model)

| Workflow | Main entry point | Output |
| --- | --- | --- |
| Physical connection | `ur_control.launch.py`, then `ur_moveit.launch.py` | Robot feedback and MoveIt/Servo |
| Manual teleop | `joy_node` → `logitech_servo` → Servo | Arm movement and gripper IO |
| Orange-cube recording | `recorder.launch.py` + separately started mapper | `data/dataset/staging/episodes/` |
| Experiment 2 recording | `scripts/record_experiment2_sort_fruits.sh` | `data/dataset/staging/experiment_2_sort_fruits/` |
| Conversion | `ur5_vla_dataset_recorder.convert_lerobot` | `data/dataset/lerobot/NEW_DATASET_NAME/` |
| Training/testing | Model-specific commands in chapter 5 | Separate model checkpoints, logs and environments |

**Repository and data:** the guide repository contains source and instructions. A Git clone does not include the ignored raw episodes, model weights, Conda environments, calibration measurements or ROS build/install directories. The commands using those assets target the configured lab PC. New-machine preparation is described in §1.2 and §5.1; copy the appropriate dataset/checkpoint/calibration assets separately.

## 1. Connect the UR5 to the remote PC

### 1.1 Prepare every ROS terminal

On the existing lab PC, run this block in **every** terminal used for ROS, teleop, recording or physical policy execution:

```bash
conda deactivate
export UR5_VLA_ROOT="/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation"
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/local_setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/local_setup.bash"
cd "$UR5_VLA_ROOT"
```

If environments are nested, repeat `conda deactivate` until the Conda environment is inactive. An `export` normally prints nothing: it sets a variable for later commands. Verify it with:

```bash
echo "$UR5_VLA_ROOT"
ros2 pkg prefix ur5_vla_dataset_recorder
ros2 pkg prefix robotiq_driver
/usr/bin/python3 -c 'import ur5_vla_dataset_recorder as p; print(p.__file__)'
```

The recorder must resolve into this repository's `ros2_ws`; Robotiq must resolve into its `hardware_ws`. ROS uses `/usr/bin/python3` and ROS 2 Humble's Python 3.10 bindings. Model launchers select their own Python environments. Keep the external drive mounted.

### 1.2 Physical-arm installation and workspace build — new workstation only

Our platform is Ubuntu 22.04 / ROS 2 Humble / **UR5 (`ur_type:=ur5`)**. `hardware_ws` provides Robotiq/serial support and simulation assets; the physical UR driver and MoveIt packages are installed under `/opt/ros/humble`. Isaac launch files are separate from physical-arm bring-up.

For a separate documentation/source checkout:

```bash
git clone https://github.com/shafakatmasud/Ur5-vla-complete-guide.git
cd Ur5-vla-complete-guide
export UR5_VLA_ROOT="$PWD"
```

Use this checkout's path in subsequent terminals on that machine. On the existing lab PC, keep the path in §1.1. Do not layer both checkouts' workspaces in the same shell.

Install ROS Humble using its [official Ubuntu installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html), then, with the ROS apt repository configured:

```bash
sudo apt update
sudo apt install \
  ros-humble-ur ros-humble-moveit ros-humble-moveit-servo \
  ros-humble-joy ros-humble-realsense2-camera ros-humble-rqt-image-view \
  ros-humble-cv-bridge ros-humble-ur-msgs ros-humble-ur-dashboard-msgs \
  ros-humble-control-msgs ros-humble-controller-manager-msgs \
  ros-humble-moveit-msgs ros-humble-trajectory-msgs \
  python3-opencv python3-numpy python3-yaml \
  python3-colcon-common-extensions python3-rosdep zip unzip
```

If rosdep has never been initialized on the new machine:

```bash
sudo rosdep init
rosdep update
```

Resolve dependencies and build, with Conda inactive:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths "$UR5_VLA_ROOT/hardware_ws/src" \
  "$UR5_VLA_ROOT/ros2_ws/src" --ignore-src --rosdistro humble -y
cd "$UR5_VLA_ROOT/hardware_ws"
colcon build --symlink-install
source install/local_setup.bash
cd "$UR5_VLA_ROOT/ros2_ws"
colcon build --symlink-install --packages-select ur5_vla_dataset_recorder
source install/local_setup.bash
```

Build/install/log directories are generated locally. The installed robot-specific URCap must match the pendant's PolyScope version; the [official UR driver robot setup](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_robot_driver/doc/installation/robot_setup.rst) describes installation and program creation.

### 1.3 Ethernet, pendant and External Control

| Item | Our setting |
| --- | --- |
| Robot control-box IP | `10.0.1.38` |
| Remote PC / reverse IP | `10.0.1.42` |
| Pendant External Control remote host | `10.0.1.42` |
| Pendant script receive port | `50002` |
| Arm type | `ur5` |
| Lab testing speed slider | At most `20%` |

Connect the control box and PC over Ethernet. The PC's robot-facing interface must actually own `10.0.1.42`; changing Wi-Fi can change routing. Check rather than changing the addresses arbitrarily:

```bash
ip -4 addr
ip route get 10.0.1.38
ping -c 3 10.0.1.38
```

On the pendant, open the saved program containing **External Control**, using remote host `10.0.1.42` and port `50002`. Start the PC driver first, then press Play on that pendant program. Restore robot safety/program state on the pendant before enabling movement.

### 1.4 Terminal A — connect the physical UR5 with our command

After §1.1:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 \
  robot_ip:=10.0.1.38 \
  reverse_ip:=10.0.1.42 \
  launch_rviz:=false \
  initial_joint_controller:=forward_position_controller \
  activate_joint_controller:=false
```

Leave this terminal running. Now play the saved External Control program on the pendant and keep the speed slider at or below 20%.

In a separate inspection terminal:

```bash
ss -ltnp | rg ':50002'
ros2 topic echo /joint_states --once
ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning '{}'
ros2 service call /dashboard_client/get_safety_mode ur_dashboard_msgs/srv/GetSafetyMode '{}'
```

A successful ping alone does not prove the External Control connection. If the pendant reports `10.0.1.42:50002: Connection refused`, verify that the correct driver is running and listening on that PC address/port. After connectivity returns, retry the pendant program; then check feedback again.

### 1.5 Terminal B — start MoveIt and Servo

After §1.1:

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 \
  launch_rviz:=true \
  launch_servo:=true
```

Leave this terminal running. Activate the forward controller in the inspection terminal:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller \
  --strict
ros2 control list_controllers
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger '{}'
ros2 topic echo /servo_node/status --once
```

Servo status `0` is required by the recording profile. If a controller is already in the requested state, inspect `list_controllers` before repeating a switch. Keep MoveIt running for guarded standard-pose moves and policy offset/path checks.

### 1.6 Arm-specific calibration — installation reference

For a new physical installation, extract this arm's factory calibration:

```bash
mkdir -p "$UR5_VLA_ROOT/hardware_ws/config"
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=10.0.1.38 \
  target_filename:="$UR5_VLA_ROOT/hardware_ws/config/ur5_calibration.yaml"
```

Supply `kinematics_params_file:="$UR5_VLA_ROOT/hardware_ws/config/ur5_calibration.yaml"` to the driver once extracted, and configure a matching calibrated MoveIt description for Cartesian work. The repository currently has no `hardware_ws/config/ur5_calibration.yaml`; simulation default kinematics and camera-to-robot calibration are different assets. The daily command in §1.4 is preserved as used in our lab.

## 2. Connect and use teleop

### 2.1 Terminal C — connect the Logitech input

With chapters 1.4–1.5 running, prepare the terminal using §1.1 and enumerate devices:

```bash
ros2 run joy joy_enumerate_devices
```

Use the **Logitech Extreme 3D Pro's displayed ID** below; the ID can change after USB reconnection:

```bash
UR5_JOYSTICK_ID=0  # Replace 0 if enumeration shows another Logitech ID.
ros2 run joy joy_node --ros-args \
  -p device_id:="$UR5_JOYSTICK_ID" \
  -p deadzone:=0.1 \
  -p autorepeat_rate:=50.0
```

Leave it running. In the inspection terminal, check that moving the stick changes `/joy` and there is exactly one input publisher:

```bash
ros2 topic echo /joy --once
ros2 topic info /joy --verbose
```

`joy_node` only supplies input. The mapper in §2.3 turns that input into arm/gripper commands.

### 2.2 Move to the standard starting pose before enabling the mapper

Stop existing policy executors, control engines and teleop mappers. With no object held and the swept path clear, run:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

This is **physical arm movement**. The helper checks the path, prepares the starting pose, restores the forward controller and starts Servo. Wait for **`DATASET START READY`** before starting the mapper. This is the standard-pose command for dataset collection; Pi0 has its own wrapper in §5.4.

### 2.3 Terminal D — start the teleop mapper

For ordinary teleop or the generic cube recorder, after §1.1:

```bash
ros2 run ur5_vla_dataset_recorder logitech_servo
```

For **experiment 2**, skip this separate mapper command: §3.3's launcher starts its own mapper. It refuses an already-running mapper/recorder or policy/control engine.

### 2.4 Joystick controls

| Physical control | Function | Joy index |
| --- | --- | --- |
| Button 1 / trigger, held | Enable arm motion; release sends zero Twist | `buttons[0]` |
| Stick forward/back | Base-frame X translation | `axes[1]` |
| Stick left/right | Base-frame Y translation | `axes[0]` |
| Handle twist | Base-frame Z rotation | `axes[2]` |
| Button 5 / button 6, with trigger held | Z down / Z up | `buttons[4]` / `[5]` |
| Button 2 | Close gripper | `buttons[1]` |
| Button 3 | Open gripper | `buttons[2]` |
| Button 7 held for two seconds, trigger released | Guarded standard-pose reset | `buttons[6]` |
| Button 9 | Start episode, when recorder is running | `buttons[8]` |
| Button 10 | Abort and discard active/pending episode | `buttons[9]` |
| Button 11 | Stop and save episode with success label | `buttons[10]` |

Gripper buttons work without holding the arm trigger. For recording, first press button 3 to open the empty gripper and initialize the acknowledged gripper stream:

```bash
ros2 topic echo /gripper/state --once
ros2 topic echo /gripper/command --once
```

Gripper convention: **0 = closed, 1 = open**. The current Robotiq electric gripper is controlled through UR digital-output pins 0/1 and SetIO. The recorded state is an IO/command estimate, not measured finger force, position or object contact. Older internal strings mentioning “pneumatic” describe this command-estimate schema.

The checked-in mapper requests `0.30 m/s` linear and `0.50 rad/s` angular maximum input magnitudes in `base_link`; Servo, robot configuration and speed scaling also affect realized speed. These are constants in [logitech_servo_node.py](ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/logitech_servo_node.py), not ROS speed parameters. Check the mapper's startup message when comparing a previously adjusted installation. This documentation update does not change speed constants.

### 2.5 Stop teleop or return to trajectory control

Release the trigger, then Ctrl+C the mapper to stop teleop. Ctrl+C the `joy_node` terminal when finished. For a switch to policy execution, follow §5.2 instead of running both command sources.

If returning to the trajectory controller with teleop/policies stopped:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger '{}'
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller \
  --strict
ros2 control list_controllers
```

## 3. Prepare and collect a dataset

### 3.1 Terminal E/F — start both cameras

Use §1.1 in each terminal. Base D435i, terminal E:

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_044322072365 \
  camera_namespace:=base camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Wrist D435, terminal F:

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_827312073590 \
  camera_namespace:=wrist camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Keep both running. View each RGB feed by starting this command twice, selecting one topic in each window:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/base/camera/color/image_raw` and `/wrist/camera/color/image_raw`. Check RGB and aligned depth individually; stop each `hz` command with Ctrl+C before the next:

```bash
ros2 topic hz /base/camera/color/image_raw
ros2 topic hz /wrist/camera/color/image_raw
ros2 topic hz /base/camera/aligned_depth_to_color/image_raw
ros2 topic hz /wrist/camera/aligned_depth_to_color/image_raw
```

The camera profile is 640×480 at 30 Hz; the recorder samples synchronized observations/actions at **20 Hz**. Both depth streams are required by the supplied recording configuration, even though the current training conversion exports RGB only.

### 3.2 Orange-cube recording — generic recorder

Run chapters 1–2, both cameras, the guarded standard-pose command and **one** teleop mapper first. In the recorder terminal:

```bash
ros2 launch ur5_vla_dataset_recorder recorder.launch.py
```

In the inspection terminal, before pressing button 9:

```bash
ros2 param set /dataset_recorder task "Pick up the orange cube and place it in the bowl"
ros2 param get /dataset_recorder output_root
ros2 param get /dataset_recorder episode_subdirectory
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger '{}'
```

Expected save root: **`$UR5_VLA_ROOT/data/dataset/staging/episodes/`**. Set the task while idle; each saved episode retains the task at episode start. For a different task/output use a new subdirectory, with the recorder idle:

```bash
ros2 param set /dataset_recorder task "YOUR EXACT DEMONSTRATED TASK"
ros2 param set /dataset_recorder episode_subdirectory "staging/YOUR_NEW_EXPERIMENT/episodes"
```

These are recording metadata/path changes. They do not train or retarget an existing model.

### 3.3 Experiment 2 — “Sort the fruits,” two apples plus two distractor cubes

Use one complete episode for **both apples**: pick the first apple, place it in the bowl, continue from that placement pose to pick the second apple, place it, then save success. Leave both cubes outside the bowl. Return to standard pose **between episodes**, rather than between apples, for this demonstrated task.

Start the driver, MoveIt, cameras and one `joy_node` as above. Run the guarded standard-pose command from §2.2. Leave the separate teleop mapper and generic recorder stopped. Start this single wrapper:

```bash
bash "$UR5_VLA_ROOT/scripts/record_experiment2_sort_fruits.sh"
```

It starts **the mapper, experiment-2 recorder and per-episode telemetry monitor**. The task is already `Sort the fruits`. It checks normal robot safety state, a running pendant program, one joystick publisher and competing nodes before launching.

| Setting | Exact value |
| --- | --- |
| Recorder config | `ros2_ws/src/ur5-vla-dataset-recorder/config/recorder_experiment2_sort_fruits.yaml` |
| Task | `Sort the fruits` |
| Raw episode save root | `data/dataset/staging/experiment_2_sort_fruits/episodes/` |
| Telemetry/evidence root | `data/dataset/staging/experiment_2_sort_fruits/telemetry/` |
| Episode telemetry index | `telemetry/episodes.csv` |
| Setup/ROS logs | `data/dataset/staging/experiment_2_sort_fruits/setup/ros_logs/` |

Press **button 3** once with the empty gripper to initialize its state. Check readiness using:

```bash
ros2 param get /dataset_recorder task
ros2 param get /dataset_recorder episode_subdirectory
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger '{}'
```

Then use 9 to start, 10 to discard, 11 to save success. No chat message or attempt-number gate is required. The read-only monitor attaches to each recorder episode automatically and uses the finalized metadata for its label.

### 3.4 One episode, step by step

1. Arrange the objects, clear the arm's path, keep the bowl/cameras at their established locations, and start from the standard pose with the empty gripper open.
2. Confirm recorder `ready: true`; it needs fresh joints, actual forward-controller targets, both RGB/depth streams, gripper state/command and Servo status 0. At least 20 GiB free space is required.
3. Press physical **9** once. Wait for the recorder to accept the episode and show it as active.
4. Hold the trigger to teleoperate, use **2/3** to close/open the gripper, and perform the complete task. Release the trigger to hold position.
5. After completion press **11** once, release it, and wait for `saved success episode to ...`. If abandoning it, press **10** instead.
6. Confirm the final episode path and status, reset the scene, then return to standard pose with button 7 (trigger released, hold two seconds) before the next episode.

Prefer smooth demonstrations, varied valid object positions and consistent camera setup. Review slips, collisions, accidental joystick movements, wrong-object pickups and long unintended pauses before adding an episode to training. For experiment 2, “success” means both apples placed and cubes left untouched; it is the operator's label.

### 3.5 How to know an episode really saved

While active or awaiting finalization, the writer uses **`.episode_YYYYMMDD_HHMMSS.incomplete`**. A completed save becomes **`episode_YYYYMMDD_HHMMSS`** with `metadata.yaml` and `frames.jsonl`. A hidden incomplete directory is not a finalized training episode; do not rename it manually.

Check live status:

```bash
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger '{}'
```

For experiment 2, list saved metadata and inspect a specific episode:

```bash
UR5_RAW_ROOT="$UR5_VLA_ROOT/data/dataset/staging/experiment_2_sort_fruits/episodes"
find "$UR5_RAW_ROOT" -maxdepth 2 -name metadata.yaml -path '*/episode_*/metadata.yaml' | sort
UR5_EPISODE=episode_YYYYMMDD_HHMMSS  # Replace with the saved episode name.
cat "$UR5_RAW_ROOT/$UR5_EPISODE/metadata.yaml"
wc -l "$UR5_RAW_ROOT/$UR5_EPISODE/frames.jsonl"
```

`metadata.yaml` must contain the final `outcome: success` and frame information. File-browser refresh alone is not a save acknowledgment. A new button press requires release of the previous button; inspect `/joy` if button 11 is not seen. If an error stops recording, inspect the terminal and status before starting another episode.

The equivalent service workflow is useful for diagnosis. The **service** `save_success` requires a stopped pending episode, so stop first; button 11 combines both steps:

```bash
# Alternative to button 9:
ros2 service call /dataset_recorder/start_episode std_srvs/srv/Trigger '{}'

# Alternative to button 11, once the task is complete:
ros2 service call /dataset_recorder/stop_episode std_srvs/srv/Trigger '{}'
ros2 service call /dataset_recorder/save_success std_srvs/srv/Trigger '{}'

# Alternative to button 10, for an active episode:
ros2 service call /dataset_recorder/stop_episode std_srvs/srv/Trigger '{}'
ros2 service call /dataset_recorder/discard_episode std_srvs/srv/Trigger '{}'
```

Only run the alternative matching the current episode state. `save_failure` also exists for retaining a stopped failed episode with a failure label; the normal button-10 route discards its training recording.

### 3.6 What telemetry and images are saved

Raw `frames.jsonl` records frame timing, measured joints, absolute command targets, gripper estimate/command and references to both camera views and depth. Raw metadata records task, joint order, units, source topics, outcome and dropped samples. The supplementary experiment-2 monitor adds:

| Per-episode evidence | Contents |
| --- | --- |
| `observations.jsonl` | Timestamped topic evidence, joints, arm targets, IO, Servo/safety/speed messages, TCP pose when available, warnings/errors |
| `summary.json`, `trial.json`, `analysis.txt` | Outcome, measured duration, path length, command/measured gaps and capture limits |
| `video/base.avi`, `video/wrist.avi` | Both camera videos, nominal playback 10 FPS |
| `camera_snapshots/` | Base/wrist JPEGs roughly once per second |
| `depth/` | Sampled depth evidence |
| `configuration/` | Source hashes/text, recorder metadata, camera intrinsics and available configuration |
| `telemetry/episodes.csv` | Session episode index |

The monitor polls recorder state every 0.25 s, so its capture may start after raw recording begins. Raw dataset timestamps retain the original onset; video-frame source timestamps are in `observations.jsonl`. IO does not prove contact, and TCP path length includes sensor noise. Aborted episodes can retain supplementary evidence even when their raw training episode is discarded.

Inspect experiment-2 monitoring:

```bash
cat "$UR5_VLA_ROOT/data/dataset/staging/experiment_2_sort_fruits/telemetry/status.json"
cat "$UR5_VLA_ROOT/data/dataset/staging/experiment_2_sort_fruits/telemetry/episodes.csv"
```

### 3.7 Data directory map

```text
data/
├── dataset/
│   ├── staging/                         # Raw demonstrations and session evidence
│   │   ├── episodes/                    # Generic/orange-cube recordings
│   │   ├── experiment_2_sort_fruits/
│   │   │   ├── episodes/                # Two-apple raw recordings
│   │   │   ├── telemetry/               # Per-episode evidence and episodes.csv
│   │   │   └── setup/                   # Recorder launch/log evidence
│   │   └── archives/                    # Operator-created raw ZIP backups
│   ├── quarantine/                      # Excluded/problem recordings for review
│   └── lerobot/                         # Converted complete datasets and packages
│       └── ur5_cube_110ep_v3/            # Reviewed experiment-1 training dataset
├── models/
│   ├── checkpoints/                     # Model weights, optimizer state and run logs
│   ├── evaluation/                      # Checks, trials, images and calibration evidence
│   ├── training_environments/           # Separate model envs/caches/dataset copies
│   └── resource_locks/                  # Cooperative GPU reservation files, when used
├── huggingface_cache/                   # Existing SmolVLA/model cache
└── hf_lerobot_cache/                    # Existing LeRobot cache
```

`staging`, `quarantine`, and `lerobot` are the three dataset categories. `quarantine` is excluded from routine conversion; `staging` is still subject to review. `evaluation` contains existing experiment evidence and is **not empty**. The map describes roles; generated subdirectories appear when their workflows run. Pi0's isolated runtime also has its own `experiments/pi0/data/` tree. Large data, caches, environments and weights are ignored by Git.

## 4. Zip the recordings and convert to LeRobot

### 4.1 Choose the reviewed episodes and task

Stop/finalize the episode first. Conversion reads **finalized successful raw directories**; it does not consume a ZIP directly. Use a fresh conversion terminal. This environment is independent of the ROS terminal setup:

```bash
export UR5_VLA_ROOT="/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation"
UR5_PACKAGE="$UR5_VLA_ROOT/ros2_ws/src/ur5-vla-dataset-recorder"
UR5_CONDA=/home/grasplab/miniconda3/bin/conda
UR5_RAW_ROOT="$UR5_VLA_ROOT/data/dataset/staging/experiment_2_sort_fruits/episodes"
UR5_DATASET_NAME=experiment2_sort_fruits_v1
UR5_TASK="Sort the fruits"
UR5_EPISODES=(episode_YYYYMMDD_HHMMSS episode_YYYYMMDD_HHMMSS)
```

Replace both episode placeholders with actual reviewed names. For an orange-cube conversion use `staging/episodes`, a new dataset name and the exact cube task instead. For a long reviewed list:

```bash
mapfile -t UR5_EPISODES < /absolute/path/to/reviewed_episode_names.txt
```

That text file contains one finalized episode directory name per line, with no comments or blank lines. Review the list before running anything below.

### 4.2 Make a raw ZIP backup

These commands archive only that explicit selection, with each episode directory kept intact. They create the ZIP outside the raw episode root:

```bash
UR5_ARCHIVE_DIR="$UR5_VLA_ROOT/data/dataset/staging/archives"
mkdir -p "$UR5_ARCHIVE_DIR"
UR5_RAW_ZIP="$UR5_ARCHIVE_DIR/${UR5_DATASET_NAME}_raw_$(date +%Y%m%d_%H%M%S).zip"
(
  set -e
  cd "$UR5_RAW_ROOT"
  test "${#UR5_EPISODES[@]}" -gt 0
  for UR5_EPISODE in "${UR5_EPISODES[@]}"; do
    test -f "$UR5_EPISODE/metadata.yaml"
    test -f "$UR5_EPISODE/frames.jsonl"
  done
  test ! -e "$UR5_RAW_ZIP"
  zip -r "$UR5_RAW_ZIP" "${UR5_EPISODES[@]}"
)
unzip -t "$UR5_RAW_ZIP"
sha256sum "$UR5_RAW_ZIP" > "$UR5_RAW_ZIP.sha256"
```

The ZIP backs up raw data. The converter still receives `--input-root "$UR5_RAW_ROOT"`. To convert an archived selection on another machine, extract into a new directory, then point the converter at it:

```bash
UR5_EXTRACT_ROOT="$UR5_VLA_ROOT/data/dataset/staging/${UR5_DATASET_NAME}_restored"
(
  set -e
  test ! -e "$UR5_EXTRACT_ROOT"
  mkdir -p "$UR5_EXTRACT_ROOT"
  unzip "$UR5_RAW_ZIP" -d "$UR5_EXTRACT_ROOT"
)
UR5_RAW_ROOT="$UR5_EXTRACT_ROOT"
```

Keep supplementary `telemetry/` separately if you want its videos and analysis; the raw ZIP selection above covers training episodes only.

### 4.3 Verify the conversion environment

```bash
"$UR5_CONDA" run --no-capture-output -n pi0 \
  python -c 'import sys, lerobot; import importlib.metadata as m; print(sys.executable); print(lerobot.__file__); print(m.version("lerobot"))'
```

**`pi0` here is the Conda environment name**, because it contains the compatible LeRobot/video dependencies. It does not make the converted dataset Pi0-only. The lab environment uses LeRobot 0.5.2 with its existing editable source under `dump/vla_script_training/lerobot`. The active converter is the ROS package's `convert_lerobot.py`; legacy CSV converters under `dump/` are a different format.

### 4.4 Convert raw episodes to a new LeRobot dataset

Use the variables set in §4.1:

```bash
PYTHONPATH="$UR5_PACKAGE" "$UR5_CONDA" run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.convert_lerobot \
  --input-root "$UR5_RAW_ROOT" \
  --output-root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --task "$UR5_TASK" \
  --episodes "${UR5_EPISODES[@]}" \
  --depth none \
  --storage video \
  --action-mode absolute
```

The converter validates all selected raw recordings, including full media decoding, **before creating output**. It refuses existing destinations by default. Use a new version name for another conversion; `--overwrite` deletes the previous destination and is not part of the routine commands.

| Result / option | Meaning |
| --- | --- |
| `observation.images.base`, `.wrist` | RGB camera features, 640×480 |
| `observation.state` | Six measured UR5 joints in radians + gripper estimate |
| `action` | Six absolute commanded joint positions in radians + binary gripper |
| Joint order | Shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3 |
| Gripper | 0 closed / 1 open |
| Frequency | 20 Hz; original timing also retained as `source_episode_time_s` |
| `--depth none` | Current RGB training workflow; raw depth is preserved |
| `--depth meters` | Optional depth/masks for compatible models |
| `--storage video` | RGB videos plus data/metadata; `image` is the alternative |
| `--action-mode absolute` | Current trained policies' contract |
| `--action-mode delta` | Alternate arm action `q_command - q_measured`; requires a matching separately trained/executed policy |

Keep `meta/`, `data/`, `videos/` and `meta/ur5_vla_conversion.json` together. `--repo-id local/...` names the local dataset; this command does not upload to Hugging Face. Omitting `--episodes` selects all finalized directories and is not equivalent to our reviewed selection.

### 4.5 Validate the converted dataset

```bash
PYTHONPATH="$UR5_PACKAGE" "$UR5_CONDA" run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.validate_lerobot \
  --root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --expected-episodes "${#UR5_EPISODES[@]}" \
  --task "$UR5_TASK" \
  --video-backend pyav \
  --report "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME/validation_report.txt"
```

A passing validator prints `PASS`; failures return nonzero and write a report. It checks every state/action row and selected video frames from both views. Add `--expected-frames NUMBER` when known. Conversion output metadata has a historical compatibility label distinct from the environment's installed LeRobot version.

The already-converted experiment-1 dataset is `ur5_cube_110ep_v3`: **110 reviewed episodes, 285,204 frames**, absolute actions. Its dated selection wrapper is:

```bash
bash "$UR5_VLA_ROOT/docs/episode_audit_20261002/convert_reviewed_episodes.sh" --help
bash "$UR5_VLA_ROOT/docs/episode_audit_20261002/convert_reviewed_episodes.sh" combined
```

`combined` refuses to overwrite that existing dataset. It reconstructs the historical selection, not new experiment-2 episodes. Use the explicit new-name workflow above for new data.

### 4.6 Package the converted dataset for Colab / persistent storage

Our training notebooks accept the complete dataset folder or a verified **uncompressed `.tar` plus checksum**. After successful conversion/validation:

```bash
PYTHONPATH="$UR5_PACKAGE" "$UR5_CONDA" run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.package_dataset \
  --dataset-root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --archive "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME.tar" \
  --repo-id "local/$UR5_DATASET_NAME"

(
  cd "$UR5_VLA_ROOT/data/dataset/lerobot"
  sha256sum -c "$UR5_DATASET_NAME.tar.sha256"
)
```

Packaging creates the TAR, checksum, manifest, metadata checksums and a verification report. It extracts into temporary storage and validates the extracted copy, so allow space for the archive and a second dataset copy. It refuses existing packages by default.

Optional 400 MiB upload parts:

```bash
PYTHONPATH="$UR5_PACKAGE" "$UR5_CONDA" run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.split_archive \
  --archive "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME.tar" \
  --part-size-mib 400
```

If you split an archive for transfer, put all parts and the checksum files into one directory on the destination machine. Reassemble into a separate new directory, then verify:

```bash
UR5_PARTS_DIR=/absolute/path/to/transferred_parts
UR5_REASSEMBLED_DIR=/absolute/path/to/new_reassembled_directory
(
  set -e
  cd "$UR5_PARTS_DIR"
  sha256sum -c "$UR5_DATASET_NAME.tar.parts.sha256"
  test ! -e "$UR5_REASSEMBLED_DIR"
  mkdir -p "$UR5_REASSEMBLED_DIR"
  cat "$UR5_DATASET_NAME.tar".part-[0-9][0-9][0-9] > "$UR5_REASSEMBLED_DIR/$UR5_DATASET_NAME.tar"
  cp "$UR5_DATASET_NAME.tar.sha256" "$UR5_REASSEMBLED_DIR/"
  cd "$UR5_REASSEMBLED_DIR"
  sha256sum -c "$UR5_DATASET_NAME.tar.sha256"
)
```

Upload the restored `.tar` and its checksum to the notebook's configured dataset path. The notebooks expect that TAR or a complete dataset folder, not raw ZIP or unreassembled parts.

For experiment 1, set `UR5_DATASET_NAME=ur5_cube_110ep_v3` and package its existing dataset the same way. Upload the folder, or `.tar` **and `.tar.sha256`**, into the notebook's configured `datasets/lerobot/` folder on Drive. Set `DRIVE_ROOT` in the notebook to the actual location; SmolVLA's older default and the later notebooks' thesis-backup default differ.

## 5. Train and run each model

### 5.1 Model boundaries, environments and shared resources

The five-model comparison uses the reviewed **experiment-1 orange-cube dataset**. The existing training launchers/notebooks target that dataset. For experiment 2, create a separately named training configuration/output and update dataset ID/path/task/expected counts; do not silently replace the experiment-1 dataset or reuse its normalization/checkpoints.

| Model | Training entry | Environment / source | Inference | Physical robot executor |
| --- | --- | --- | --- | --- |
| SmolVLA | A100 notebook | Existing `pi0` environment; original `scripts/` | Port 8765, completed 18k checkpoint | Available, §5.3 |
| Pi0 | Local FSDP or A100 notebook | Training `pi0`; runtime `pi0_ur5_runtime`; `experiments/pi0/` snapshot | Port 8766, completed 20k checkpoint | Available, §5.4 |
| Pi0.5 | `experiments/pi05/train.sh` or A100 notebook | `pi05_ur5`, private `experiments/pi05/lerobot/` | Completed 20k checkpoint | Separate live adapter still required, §5.5 |
| OpenVLA | Isolated A100 notebook | `/content/openvla_ur5/env`; `experiments/openvla_a100/` | Completed 20k LoRA adapter; saved-image inference | Separate live adapter still required, §5.6 |
| ACT | `experiments/act/start.sh` | `data/models/training_environments/act/env`; own source/dataset copy | Dedicated run targeting 20k | Separate live adapter still required, §5.7 |

ACT is a task-specific imitation baseline without language conditioning. OpenVLA uses base RGB + language only. SmolVLA/Pi0/Pi0.5 use two RGB views + state + language; ACT uses two views + state. Their training losses and input conditions are not interchangeable success metrics.

Use each model's own Python/source/checkpoint/normalization. Run launchers with `bash`, not `source`, so their environment variables remain in the child process. New work should stay in that model's directory. The shared robot accepts one command source at a time.

Inspect isolation and GPUs without loading a model:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/check_isolation.py"
cat "$UR5_VLA_ROOT/experiments/models.json"
nvidia-smi
```

The cooperative GPU guard reserves named physical GPUs and refuses busy devices; it never kills another model. Pi0/Pi0.5 local training use GPUs 0,1,2. ACT uses GPU 2. They must not overlap. Example for a guarded job:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/experiments/pi05/train.sh" --probe
```

Legacy unwrapped commands still exist and bypass cooperative reservations. Before any physical GPU-0 inference, inspect current GPU use and checkpoint/pause the training job using its own helper. Pi0's wrapper specifically pauses an active Pi0.5 job; other launchers do not imply that behavior.

**Dependencies on this lab PC:** conversion/SmolVLA/Pi0 training already have the `pi0` environment; Pi0 testing and Pi0.5 have separate prepared environments. The Conda folders are not in Git. ACT's setup clones the existing dependency environment into its own prefix and installs only its private source. OpenVLA's notebook creates its own cloud environment. For fresh cloud training, use the pinned setup cells in the linked notebook, rather than installing a new LeRobot version into the lab's working environments. Environment snapshots are in `experiments/pi0/environment_snapshot.txt` and `experiments/act/requirements.lock.txt`.

W&B authentication is per environment/account. For an existing local environment that is not logged in, run its own hidden-input login once, for example:

```bash
/home/grasplab/miniconda3/envs/pi0/bin/python -m wandb login
/home/grasplab/miniconda3/envs/pi05_ur5/bin/python -m wandb login
```

Enter the key at the prompt; do not put it into committed files or shell command text. ACT normally inherits the existing account login, and its own environment can use the same `python -m wandb login` command after setup. Colab secret names are `WANDB_API_KEY` and optionally `HF_TOKEN`, with Notebook access enabled. The OpenVLA notebook reads these directly; the other notebooks include their own login/setup cells.

### 5.2 Common physical-test preparation and fault recovery

For **SmolVLA or Pi0**, keep the driver, MoveIt and both cameras from chapters 1 and 3 running. Stop the dataset recorder, telemetry recording launcher and teleop mapper before changing to policy control. `joy_node` alone may remain. No training process should occupy the inference GPU.

Start from an empty open gripper and a reset scene, with the arm's path clear. Before starting a policy/control engine:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
ros2 control list_controllers
ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger '{}'
```

For **Pi0**, use its own `standard_pose.sh` after its preparation instead; it manages the Pi0 engine/Servo handoff. Keep the pendant speed at or below 20% during these trials.

A policy engine publishes filtered absolute joint targets at 125 Hz; Servo teleop also writes to the forward controller. Stop competing command sources before policy execution. When changing models, Ctrl+C the old policy/server, stop its engine, then use the next model's preparation. Pi0's dedicated engine can be stopped with:

```bash
systemctl --user stop ur5-pi0-control-engine.service
```

The standalone SmolVLA engine is stopped with Ctrl+C in its terminal. Global engine locking prevents two compliant engine processes; it cannot arbitrate arbitrary other direct joint publishers.

If startup is blocked, inspect first:

```bash
ros2 node list
ros2 topic info /control_abstraction_layer --verbose
ros2 topic info /forward_position_controller/commands --verbose
ros2 service call /control_engine/status std_srvs/srv/Trigger '{}'
ros2 topic echo /joint_states --once
ros2 topic echo /servo_node/status --once
```

After resolving the cause, with the old policy stopped, the engine's reset service is:

```bash
ros2 service call /control_engine/reset std_srvs/srv/Trigger '{}'
```

Reset must return `success=True` and requires fresh measured joints, Servo-stop acknowledgment and no competing/armed policy. Do not loop resets through protective stops, stale feedback, tracking errors or missing planning services. Keep MoveIt running for offset/FK/state-validity checks; a timeout is a blocked check, not permission to bypass it. Software stop commands are not emergency stops.

### 5.3 SmolVLA — train, check and run

#### 5.3.1 Train on A100 using the existing notebook

Open [the SmolVLA A100 notebook](notebooks/train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb) in Colab. Select an A100 runtime, set `DRIVE_ROOT`/dataset paths in configuration cell 2, then run cells in order: pinned dependency setup → dataset hash/shape checks → W&B login → batch benchmark → training budget → training → verified Drive export.

The notebook benchmarks batches 16–128 and derives the step budget from eight dataset passes. Our deployed run selected batch 128 / **18,000 steps**. Its training command is printed and saved in `resolved_training_config.json`. In the prepared **cloud training environment**, the corresponding explicit command is:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$TRAIN_STEPS" \
  --dataset.repo_id="$REPO_ID" --dataset.root="$LOCAL_DATASET" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed=42 \
  --wandb.enable=true --wandb.project="$WANDB_PROJECT" \
  --save_checkpoint=true --eval_freq=0 \
  --steps="$TRAIN_STEPS" --save_freq="$SAVE_FREQ" --log_freq=10 \
  --output_dir="$TRAIN_OUTPUT" --job_name="$RUN_NAME"
```

These variables are the notebook's resolved values; use its training cell to retain GPU logs, checkpoint copying and export verification. Do not run this again alongside an already-running training cell. The full exact code-cell commands for all supplied notebooks are also preserved in [notebook command reference](docs/reference/notebook_training_commands.txt).

To resume with the upstream trainer, a **complete training checkpoint** is required, including `training_state/`, not only an inference export:

```bash
# In the same prepared cloud environment, use your actual complete checkpoint path.
UR5_SMOL_CHECKPOINT=/absolute/path/to/run/checkpoints/018000
lerobot-train \
  --config_path="$UR5_SMOL_CHECKPOINT/pretrained_model/train_config.json" \
  --resume=true
```

Inspect the saved output/dataset paths before resuming in a replacement runtime; the older SmolVLA notebook's new-run cells deliberately reject an existing run. Use the saved training config and complete checkpoint for recovery.

#### 5.3.2 Existing trained checkpoint and no-motion tests

Deployed model:

```text
data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model/
```

Keep weights, `config.json`, preprocessor/postprocessor files and tokenizer assets together. The standalone executor selects this model using `MODEL_PATH` in `scripts/policy_executor.py`; it has no `--checkpoint` argument. The control engine has no model/dataset path.

Saved-observation test, without ROS commands:

```bash
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" offline
```

It checks three samples per episode, writes CSV/JSON to `data/models/evaluation/smolvla_110ep_18000/`, and enforces its prediction envelope. Optional live-camera shadow check, in separate terminals:

```bash
# Server terminal; no robot publishers.
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" server
```

```bash
# ROS shadow-client terminal; no arm/gripper commands.
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" shadow
```

Stop both before physical testing; the physical executor starts its own server on port 8765. The installed `ros2 run ur5_vla_dataset_recorder policy_executor` is a shadow-only program; the physical executor is the standalone script below.

#### 5.3.3 Physical execution — engine terminal

Finish §5.2, then:

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 control_engine.py --speed-scale 0.5 \
  --bowl-release-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/bowl_release_calibration.json"
```

This uses the existing taught bowl-region release supervision. Keep the bowl at its calibrated position. If intentionally testing raw model gripper decisions, omit `--bowl-release-file` and label that execution condition separately. The engine waits for fresh feedback and Servo-stop acknowledgment, smooths targets and serializes gripper IO. Leave the teleop mapper stopped.

#### 5.3.4 Physical execution — policy terminal, same command each attempt

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --task "Pick up the orange cube and place it in the bowl" \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous \
  --prepare-episode \
  --speed-scale 0.5 \
  --actions-per-prediction 50
```

This is physical execution. `--prepare-episode` opens the gripper at start, so begin with no object held. At scale 0.5 the policy issues targets at 10 Hz, executes 50 actions before the next chunk, and the engine smooths output at 125 Hz. The saved Cartesian offset is applied to arm targets; the continuous-offset profile supplies a default 25 mm carry lift unless explicitly overridden. `--carry-lift-mm 0` disables that lift for a separately specified condition.

For the explicitly assisted **planned placement** condition, use this policy command with the same bowl-enabled engine:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/policy_executor.py" \
  --task "Pick up the orange cube and place it in the bowl" \
  --planned-transport-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/bowl_release_calibration.json" \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 50
```

Planned transport supplies its own lift/route after stable closed IO; it is assisted placement, and closed IO alone does not prove the object was grasped. Existing blue-cube/orange-fruit wrappers and experimental offsets remain in source and the historical operations reference. Task text can be changed with `--task`; that alone does not establish unseen-task generalization.

Offset coordinates, limits and freshness/path gates are implemented by imported helpers `grasp_offset.py`, `trajectory_offset.py` and associated modules. Do not launch those helpers separately or infer base-frame directions from wrist-image “left/right” without calibration. Keep calibration files for their measured setup. For full option descriptions without execution:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/policy_executor.py" --help
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/control_engine.py" --help
```

#### 5.3.5 End a test / return to teleop

Ctrl+C the policy, wait for its server to exit, then Ctrl+C its engine. Secure any held object before opening or resetting. Reset the scene and use §2.2, then restart the mapper for teleop. A fault stops output and requires inspection/reset; do not increase watchdog limits to hide command gaps.

### 5.4 Pi0 — separate training and physical testing

#### 5.4.1 Local training, W&B and resume

The existing lab launcher uses GPUs 0–2, FSDP, FP16 mixed precision, frozen vision-language backbone and trained action expert/projections. Batch 1 × 3 GPUs × accumulation 4 gives effective batch 12; target 20,000 updates. It reads `data/dataset/lerobot/ur5_cube_110ep_v3` and uses the already-cached pinned `lerobot/pi0_base` weights. No robot commands are created by training.

Run one operation at a time, after freeing these GPUs:

```bash
# Real two-update loading/training probe.
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh" --probe

# New 20,000-step run, unique timestamped output.
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh"
```

The launcher prints the new output path and appends its logs to `terminal.log`. For the existing completed run:

```bash
UR5_PI0_RUN="$UR5_VLA_ROOT/data/models/checkpoints/experiment1_pi0_local/ur5_cube_110ep_pi0_expert_20261019_210719"
cat "$UR5_PI0_RUN/wandb_run.json"
tail -f "$UR5_PI0_RUN/terminal.log"
```

Recovery command for an interrupted run (replace the directory if recovering a new run):

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh" --resume "$UR5_PI0_RUN"
```

Resume restores the latest complete model/optimizer/scaler/sampler state and W&B run, within its saved step budget. A completed 20k run does not need another launch. Metrics are in `metrics.jsonl`; full checkpoints include `pretrained_model/` and training state. Older checkpoints' RNG completeness is documented in the historical reference.

The lab also has a Pi0 watchdog timer. Inspect its schedule/history with:

```bash
systemctl --user list-timers ur5-pi0-training-watchdog.timer
journalctl --user -u ur5-pi0-training-watchdog.service -n 20 --no-pager
```

Before deliberately pausing/switching GPU jobs, disable its automatic recovery if it is enabled:

```bash
systemctl --user disable --now ur5-pi0-training-watchdog.timer
```

Its legacy recovery service is independent of the new cooperative GPU wrapper. Installation/recovery details remain in the historical reference; enabling a watchdog is an explicit training-management choice.

#### 5.4.2 Alternative A100 notebook

Use [Pi0 A100 notebook](notebooks/train_ur5_cube_110ep_v3_pi0_a100_colab.ipynb). Configure dataset/Drive paths, A100 runtime, `FINETUNE_MODE='expert'`, `TRAIN_STEPS=20_000`, W&B and persistent output. Run cells in order; they install the pinned stack, verify data/pretrained loading, benchmark batches, train and export. This is a separate cloud recipe from local FP16/FSDP training. The exact policy/dataset/training command lists are in [notebook commands](docs/reference/notebook_training_commands.txt).

#### 5.4.3 Physical Pi0 preparation and standard pose

Pi0 uses **only its isolated runtime**: `experiments/pi0/`, Python environment `pi0_ur5_runtime`, private source, checkpoint and calibration copies, port **8766**. SmolVLA remains on 8765 with its original sources.

Start the common hardware/cameras, stop teleop, old policies and old engines, then:

```bash
export PI0_ROOT="$UR5_VLA_ROOT/experiments/pi0"
bash "$PI0_ROOT/prepare.sh"
/usr/bin/python3 "$PI0_ROOT/check.py"
```

Preparation starts the dedicated Pi0 engine, waits for Servo-stop acknowledgment and fresh joints, requests OPEN and waits for acknowledged IO. The existing workstation service is `ur5-pi0-control-engine.service`. Inspect it with:

```bash
systemctl --user status ur5-pi0-control-engine.service --no-pager
journalctl --user -u ur5-pi0-control-engine.service -f
```

For the starting pose, stop the Pi0 policy first, empty the gripper and run:

```bash
bash "$PI0_ROOT/standard_pose.sh"
```

This manages its own engine stop, copied guarded pose move, forward-controller/Servo handoff, engine restart and empty-gripper opening. Wait for a successful readiness result.

Optional live-camera read-only check in separate terminals:

```bash
bash "$PI0_ROOT/server.sh"
```

```bash
/usr/bin/python3 "$PI0_ROOT/check.py" --shadow --samples 5
```

Stop that optional server before the physical run. A separate source clone needs the Pi0 runtime environment, checkpoint/calibration assets and user-service setup; the existing lab PC already has these. [Pi0 command reference](experiments/pi0/COMMANDS.txt) describes the installed runtime.

#### 5.4.4 Physical Pi0 run — same command every attempt

```bash
bash "$PI0_ROOT/run.sh"
```

The wrapper pauses an active Pi0.5 training job, verifies GPU 0 availability, prepares the engine and starts its own server. It defaults to the trained orange-cube task, speed scale 0.5 and **20 actions per prediction**. Pi0 has its own arm averaging/filtering and fixed-bowl gripper-hold/release supervision. The first accepted close is held until bowl release; these are execution guards, not proof that the model learned release. It does not use the SmolVLA grasp offset or planned carry.

Logs: `experiments/pi0/data/models/evaluation/pi0_tests/`; read-only reports: `.../pi0_checks/`. Ctrl+C ends the policy and its model server. For another trial, reset the scene, optionally run its standard-pose wrapper, then repeat `run.sh`. Logs alone are not a full trial telemetry/video recorder.

To finish Pi0 testing:

```bash
systemctl --user stop ur5-pi0-control-engine.service
```

Resume Pi0.5 only if that job is incomplete and GPUs 0–2 are free, using §5.5. This service stop does not stop any training process.

### 5.5 Pi0.5 — isolated training, resume and checkpoint inspection

Own source: `experiments/pi05/`; environment: `/home/grasplab/miniconda3/envs/pi05_ur5`; own caches: `data/models/training_environments/pi05/`. The local recipe uses expert/projection finetuning, frozen backbone, FSDP/FP16 on GPUs 0–2, effective batch 12, **20,000 updates**, saved every 250. Pi0.5 has its own quantile normalization and processor assets.

With its existing environment/base-weight cache prepared, and all three GPUs free:

```bash
# Probe.
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/experiments/pi05/train.sh" --probe

# New run.
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/experiments/pi05/train.sh"
```

Existing run, metrics and W&B:

```bash
UR5_PI05_RUN="$UR5_VLA_ROOT/data/models/checkpoints/experiment1_pi05_local/ur5_cube_110ep_pi05_expert_20261029_135452"
cat "$UR5_PI05_RUN/wandb_run.json"
cat "$UR5_PI05_RUN/training_config.json"
tail -f "$UR5_PI05_RUN/terminal.log"
```

For a running Pi0.5 job, request its own checkpoint/pause and wait for workers to exit:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/pi05/pause.py"
systemctl --user status ur5-pi05-training.service --no-pager
nvidia-smi
```

For a genuinely interrupted run, resume its newest complete checkpoint:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/model_job.py" --gpus 0,1,2 -- \
  bash "$UR5_VLA_ROOT/experiments/pi05/train.sh" --resume "$UR5_PI05_RUN"
```

The workstation's legacy background-service helper is `bash "$UR5_VLA_ROOT/experiments/pi0/resume_pi05.sh"`. It is not a substitute for checking ACT's GPU-2 reservation. Prefer the guarded foreground command above when starting a job alongside the newer comparison tooling.

A100 alternative: [Pi0.5 notebook](notebooks/train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb), with its own setup, paths, model revision, normalization, W&B and checkpoint/export code. Its exact training arguments are in [notebook commands](docs/reference/notebook_training_commands.txt).

Existing trained export:

```bash
UR5_PI05_MODEL="$UR5_PI05_RUN/checkpoints/020000/pretrained_model"
ls -lh "$UR5_PI05_MODEL/model.safetensors"
cat "$UR5_PI05_MODEL/config.json"
```

**Physical execution status:** a separate Pi0.5 ROS inference/execution adapter is not implemented in this repository. There is no verified Pi0.5 `run.sh` or inference port yet. Do not substitute these weights into Pi0/SmolVLA launchers: architecture, state/action normalization, precision and processors need their own validated loader and no-motion checks first.

### 5.6 OpenVLA — isolated A100 training and saved-image inference

#### 5.6.1 Notebook setup and training

Use the notebook matching the actual GPU:

- [A100 40 GB notebook used for the completed run](notebooks/train_ur5_cube_110ep_openvla_a100_40gb.ipynb): batch 16 × accumulation 1.
- [A100 80 GB notebook](notebooks/train_ur5_cube_110ep_openvla_a100_80gb.ipynb): batch 4 × accumulation 4; checks for an actual 80 GB allocation.

Open the notebook in Colab and run its numbered cells in order:

| Cell | Action |
| --- | --- |
| 1 | Inspect assigned GPU/runtime |
| 2 | Mount Drive, choose dataset and persistent output paths / run name |
| 3 | Unpack embedded reviewed source |
| 4 | Create isolated Python 3.10 environment and install pinned packages |
| 5 | Stage/verify dataset and prepare cached base-camera examples |
| 6 | Read `WANDB_API_KEY` / optional `HF_TOKEN`, verify W&B login |
| 7 | Two actual updates/save test, then a fresh-process resume test |
| 8 | Start or resume full training |
| 9 | Inspect status, checkpoint and dashboard link |

The source uses original **OpenVLA-7B**, base camera + language, **LoRA rank 32**, BF16, constant learning rate `5e-4`, 20,000 steps, checkpoint every 250, latest two complete checkpoints retained. Actions are our six absolute UR5 joint radians + gripper, not the stock end-effector delta contract. This is not the two-camera OpenVLA-OFT variant.

After notebook setup defines `PYTHON`, `CODE`, `CACHE`, `OUTPUT` and `CHILD_ENV`, the 40 GB training command executed by cell 8 is:

```python
run([
    PYTHON, '-u', CODE / 'train.py',
    '--cache', CACHE, '--output', OUTPUT,
    '--steps', '20000', '--batch-size', '16', '--accumulate', '1',
    '--save-every', '250', '--min-gpu-memory-gib', '35',
    '--wandb-entity', 'mdshafakat-masud-ontario-tech-university',
    '--wandb-project', 'ur5-experiment1-openvla',
], env=CHILD_ENV, log=OUTPUT / 'terminal.log')
```

Use cell 8 after the probes, rather than launching a duplicate process. To pause, interrupt the training cell **once** and wait for the checkpoint-save message. To resume after an interruption, keep the same persistent `RUN_NAME` and settings, rerun setup/probes and cell 8. Source signatures must match; adapter/optimizer/scheduler/RNG/sample position restore from the newest complete checkpoint.

Completed run: `ur5_cube_110ep_openvla_lora32_b16_20000_steps_a10040_b16a1_v2`. Its checkpoint folder is:

```text
/content/drive/MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5/
  checkpoints/ur5_openvla_trained/
    ur5_cube_110ep_openvla_lora32_b16_20000_steps_a10040_b16a1_v2/
      checkpoints/020000/
```

Training assessment: [assessment.txt](experiments/openvla_a100/assessment_20261105/assessment.txt). The checkpoint is a **LoRA adapter**, so keep its pinned base-model revision, action contract and normalization assets.

#### 5.6.2 Run checkpoint inspection or saved-image prediction

In the notebook's prepared environment:

```python
CHECKPOINT = OUTPUT / 'checkpoints/020000'
run([PYTHON, CODE / 'load_checkpoint.py', '--checkpoint', CHECKPOINT], env=CHILD_ENV)
```

Optional saved **base-camera RGB** prediction, with your actual image path:

```python
BASE_RGB_IMAGE = Path('/content/base_camera_frame.jpg')
run([PYTHON, CODE / 'load_checkpoint.py', '--checkpoint', CHECKPOINT,
     '--image', BASE_RGB_IMAGE], env=CHILD_ENV)
```

The loader applies the saved action normalization/contract and prints a prediction; it publishes no ROS commands. **A physical OpenVLA executor is not implemented yet.** It needs a separate isolated inference/ROS adapter and validated live preprocessing/action timing before physical use. Full reference: [OpenVLA commands](experiments/openvla_a100/COMMANDS.txt).

### 5.7 ACT — separate local training, W&B and checkpoints

ACT has its own source archive, environment, dataset copy, caches, logs and GPU-2 reservation. It uses original LeRobot ACT with two 640×480 RGB cameras + 7D state, ImageNet ResNet18 backbone, remaining ACT weights trained from scratch, chunk size 50, action horizon 20, CVAE KL weight 10, AdamW `1e-5`, FP32, batch 16 and target **20,000 updates**. It has **no task-language input**.

Prepare the private environment/source/dataset once:

```bash
bash "$UR5_VLA_ROOT/experiments/act/setup.sh"
```

Run a real update/save test followed by a **cold-process resume** test using the same probe command twice:

```bash
bash "$UR5_VLA_ROOT/experiments/act/train.sh" --probe
bash "$UR5_VLA_ROOT/experiments/act/train.sh" --probe
```

Start/resume the fixed run in its own user service:

```bash
bash "$UR5_VLA_ROOT/experiments/act/start.sh"
systemctl --user status ur5-act-training.service --no-pager
```

`start.sh` reports an existing active service rather than starting a duplicate. It verifies the passed probe/source signatures. ACT was launched as a separate job; this guide reorganization does not restart it.

Inspect progress and W&B:

```bash
UR5_ACT_RUN="$UR5_VLA_ROOT/data/models/checkpoints/experiment1_act_local/ur5_cube_110ep_act_b16_c50_20000_steps_v1"
tail -f "$UR5_ACT_RUN/terminal.log"
cat "$UR5_ACT_RUN/wandb_run.json"
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/act/check.py"
```

Checkpoint and pause **only ACT**, then wait for its paused status/service exit:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/experiments/act/pause.py"
systemctl --user status ur5-act-training.service --no-pager
```

Resume with the same `start.sh` command. Foreground alternative after the old service exits:

```bash
bash "$UR5_VLA_ROOT/experiments/act/train.sh" --resume "$UR5_ACT_RUN"
```

The job saves every 250 updates and retains the latest two complete model/optimizer/RNG checkpoints. `check.py` inspects checkpoint status/checksums without loading the GPU model. A completed checkpoint is available for model loading; **there is no physical ACT executor or inference port yet**. A separate adapter must preserve its saved state/action normalization and action-chunk handling. Full reference: [ACT commands](experiments/act/COMMANDS.txt).

### 5.8 W&B dashboards, comparison and further reference

| Model | Dashboard / run |
| --- | --- |
| SmolVLA | Use the live URL printed by its notebook / saved training report; avoid guessing a historical run ID |
| Pi0 | `wandb_run.json` in the chosen local run |
| Pi0.5 | [Pi0.5 run](https://wandb.ai/mdshafakat-masud-ontario-tech-university/ur5-experiment1-pi05/runs/5sxfftn1) |
| OpenVLA | [Completed OpenVLA run](https://wandb.ai/mdshafakat-masud-ontario-tech-university/ur5-experiment1-openvla/runs/30396f63) |
| ACT | [ACT run](https://wandb.ai/mdshafakat-masud-ontario-tech-university/ur5-experiment1-act/runs/88916732) |

Follow loss, learning rate, gradient/AMP diagnostics, throughput and memory in each run's own project. Constant learning rates produce flat curves. These runs use all 110 episodes for training without a held-out split; training diagnostics are not physical success rates. Record offsets, bowl-release guards, planned transport, input cameras and any software interventions with physical results so the model comparison remains interpretable.

The full earlier operating notes—detailed timeout/startup diagnostics, historical cross trials, blue-cube/orange-fruit wrappers, calibration experiments, notebook history and migration records—are retained in [historical operations reference](docs/reference/operations_before_reorganization_20261105.txt). Current workflow order and commands are in this README. [Notebook code-cell command reference](docs/reference/notebook_training_commands.txt) preserves the supplied cloud setup/training commands without copying embedded source payloads. [Model inventory](experiments/models.json) identifies each source, environment and checkpoint.
