# Physical UR5 VLA implementation

Central local repository for physical UR5 vision-language-action recording, conversion, training, and evaluation at GRASP Lab, Ontario Tech University. **This is the authoritative README for current paths and operation.** Other Markdown files are package details, experiment reports, or historical records.

## Contents

- [Workspaces and hardware](#workspaces-and-hardware)
- [Location and environment](#location-and-environment)
- [Build and source](#build-and-source)
- [UR5 physical arm installation and teleoperation](#ur5-physical-arm-installation-and-teleoperation)
  - [Daily run: teleoperation and dataset recording, one step at a time](#daily-run-teleoperation-and-dataset-recording-one-step-at-a-time)
- [Directory map](#directory-map)
- [Current data and model](#current-data-and-model)
- [Dataset conversion and validation](#dataset-conversion-and-validation)
- [ROS package files](#ros-package-files)
- [Standalone scripts](#standalone-scripts)
- [Policy execution: setup, commands, offsets, and diagnostics](#policy-execution-setup-commands-offsets-and-diagnostics)
- [Typical read-only checks](#typical-read-only-checks)
- [Historical material and known limits](#historical-material-and-known-limits)
- [Verification and Git](#verification-and-git)
  - [ROS executable smoke test](#ros-executable-smoke-test-2026-10-11)

## Workspaces and hardware

This repository contains two ROS 2 workspaces with different roles:

| Workspace | Purpose | When it is used |
| --- | --- | --- |
| `hardware_ws` | Robotiq gripper driver, controller plugins, serial transport, robot description, hardware tests, and UR5 Isaac Sim support | Source before `ros2_ws`; use its Robotiq launch files only when controlling compatible serial-connected Robotiq hardware or using the simulation assets |
| `ros2_ws` | Our UR5 VLA experiment package: dataset recorder, Logitech teleoperation, guarded standard pose, conversion, validation, and policy tools | Use for the current physical recording and evaluation workflow |

`hardware_ws/src/ros2_robotiq_gripper/` contains `robotiq_driver` (a
`ros2_control` hardware-interface plugin), `robotiq_controllers` (gripper
controller plugins), `robotiq_description` (URDF, xacro, controller YAML,
and launch files), and `robotiq_hardware_tests` (driver tests).
`hardware_ws/src/serial/` is the serial communication library used by the
Robotiq driver. `hardware_ws/src/ur5_isaac_simulation/` contains the
Isaac Sim bridge, action servers, descriptions, scenes, and launch assets.
These packages provide components and launch files; most do not have a
standalone `ros2 run` executable.

The current Logitech gripper buttons in `ros2_ws` call the UR controller's
`/io_and_status_controller/set_io` service. That is a separate path from the
serial Robotiq driver in `hardware_ws`; having the Robotiq packages installed
does not mean that driver is currently controlling the physical gripper.

The operator confirmed that the installed gripper is **Robotiq electric**;
the exact model and active pendant program's force setting are not yet verified.
The current SetIO path exposes binary close/open requests, not force or speed
setpoints. Pin 0 ON requests close and pin 1 ON requests open. Successful IO
writes do not measure finger force or object contact. The recorded output
pulses return OFF after roughly two seconds; this alone does not establish
loss of grip force, since the downstream control program has not been inspected.
Check the Robotiq force/speed settings in the robot program that handles these
requests. Do not infer a force percentage from the policy's 0/1 gripper value.
Robotiq documents separate force-setting functions in its
[UR gripper control documentation](https://blog.robotiq.com/how-to-control-the-grip-force-of-the-robotiq-grippers).

Before starting a Robotiq controller, identify the connected gripper and
serial device and check for an existing controller. The relevant launch file
is `robotiq_description/launch/robotiq_control.launch.py` (its default port is
`/dev/ttyUSB0`); `hand-e_control.launch.xml` is for the Hand-E variant.

To verify which repository supplies these packages in the current terminal:

```bash
ros2 pkg prefix robotiq_driver
ros2 pkg prefix robotiq_controllers
ros2 pkg prefix robotiq_description
ros2 pkg prefix ur5_vla_dataset_recorder
```

The first three paths should be inside this repository's
`hardware_ws/install/`; the last should be inside `ros2_ws/install/`.

## Location and environment

The repository is mounted at:

```text
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
```

The user's `.bashrc` sets `UR5_VLA_ROOT` to this directory and sources ROS 2 Humble, `hardware_ws`, then `ros2_ws`. **Open a new terminal after this migration**; already-open terminals retain their previous ROS environment. Verify a fresh terminal before running a node:

```bash
echo "$UR5_VLA_ROOT"
ros2 pkg prefix ur5_vla_dataset_recorder
ros2 pkg prefix robotiq_driver
/usr/bin/python3 -c 'import ur5_vla_dataset_recorder as p; print(p.__file__)'
```

The first prefix and Python file must be under this repository's `ros2_ws`; `robotiq_driver` must be under its `hardware_ws`. To source manually:

```bash
export UR5_VLA_ROOT=/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/setup.bash"
```

ROS 2 Humble, the UR robot driver, camera drivers, system Python 3.10, and the `pi0` Conda environment are system dependencies. The `pi0` environment's editable LeRobot 0.5.2 installation now points to `dump/vla_script_training/lerobot` **inside this repository**. Conda itself remains at `/home/grasplab/miniconda3/bin/conda`. The external drive must remain mounted while using this setup. Fresh terminals also set `HF_HOME` to `data/huggingface_cache` and `HF_LEROBOT_HOME` to `data/hf_lerobot_cache`; authentication tokens were not copied.

Use `/usr/bin/python3` for ROS-facing standalone scripts; the Conda base
`python3` may be a different Python version from ROS Humble's bindings.

## Build and source

`hardware_ws` contains the copied Robotiq/Isaac source; `ros2_ws` contains the dataset recorder and policy package. Build in this order:

```bash
source /opt/ros/humble/setup.bash
cd "$UR5_VLA_ROOT/hardware_ws"
colcon build --symlink-install
source install/setup.bash
cd "$UR5_VLA_ROOT/ros2_ws"
colcon build --symlink-install --packages-select ur5_vla_dataset_recorder
source install/setup.bash
```

Six hardware workspace packages and the recorder package built successfully from copied source on 2026-10-10. Build, install, and log directories are generated output excluded from Git.

## UR5 physical arm installation and teleoperation

### Installed components and workspace roles

The physical platform uses **UR5 (`ur_type:=ur5`), ROS 2 Humble on Ubuntu
22.04, and robot IP `10.0.1.38`**. Use system Python 3.10 for ROS commands.
The following installation was inspected on 2026-10-11:

| Component | Installed version or location | Role |
| --- | --- | --- |
| `ur_robot_driver`, `ur_moveit_config`, `ur_calibration` | ROS Debian packages 2.14.0 under `/opt/ros/humble` | Physical arm connection, trajectory controllers, MoveIt configuration, calibration extraction |
| `moveit_servo` | ROS Debian package 2.5.9 | Converts teleoperation Twist commands into joint targets |
| `joy` | ROS Debian package 3.3.0 | Logitech Extreme 3D Pro input |
| `realsense2_camera` | ROS Debian package 4.58.3 | Base D435i and wrist D435 camera streams |
| `hardware_ws` | Six source packages: `serial`, `robotiq_driver`, `robotiq_controllers`, `robotiq_description`, `robotiq_hardware_tests`, `ur5_isaac_simulation` | Serial Robotiq support and Isaac simulation assets |
| `ros2_ws` | `ur5_vla_dataset_recorder` | Logitech mapper, guarded reset, synchronized recording, and policy tools |

The UR arm driver is **not contained in `hardware_ws/src/`**. Its binary
installation provides the physical arm stack. The current Robotiq electric gripper
uses UR digital outputs through SetIO; the serial Robotiq launch files are for
the separate compatible gripper setup. Isaac launch files start simulation
components and are not the physical arm bring-up procedure.

### Install dependencies on a new workstation

First install Ubuntu 22.04 and ROS 2 Humble following the
[official ROS installation instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
The UR Humble driver supports binary installation through `ros-humble-ur`;
see the [official Humble driver installation guide](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_robot_driver/doc/installation/installation.rst).
With the ROS apt repository configured, install the host dependencies:

```bash
sudo apt update
sudo apt install \
  ros-humble-ur ros-humble-moveit ros-humble-moveit-servo \
  ros-humble-joy ros-humble-realsense2-camera \
  ros-humble-cv-bridge ros-humble-ur-msgs ros-humble-ur-dashboard-msgs \
  ros-humble-control-msgs ros-humble-controller-manager-msgs \
  ros-humble-moveit-msgs ros-humble-trajectory-msgs \
  python3-opencv python3-numpy python3-yaml \
  python3-colcon-common-extensions python3-rosdep
```

If rosdep has not been initialized on that machine, run `sudo rosdep init`
once, followed by `rosdep update`. Install workspace dependencies:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths "$UR5_VLA_ROOT/hardware_ws/src" \
  "$UR5_VLA_ROOT/ros2_ws/src" --ignore-src --rosdistro humble -y
```

Set `UR5_VLA_ROOT` using [Location and environment](#location-and-environment)
first, then build and source both workspaces in the order shown in
[Build and source](#build-and-source). The existing `hardware_ws/src/serial`
supplies the serial dependency; no second checkout is needed. Isaac support
additionally needs `ros-humble-tf-transformations`, which is missing from the
current host and its package manifest; see the smoke-test limitations below.

### Robot network, External Control, and calibration

Connect the robot control box and PC over Ethernet. Configure a unique PC
address on the robot's subnet, matching the actual pendant network settings.
The current driver uses robot address `10.0.1.38` and PC reverse address
`10.0.1.42`. Check that this PC address is assigned to the chosen Ethernet
interface with `ip -4 addr`, and check reachability with `ping -c 3 10.0.1.38`.

Install the External Control URCap appropriate for the robot's PolyScope
version, reboot as directed, configure its remote host as the PC's Ethernet
IP, and save a pendant program containing the External Control node. The
installed driver provides
`/opt/ros/humble/share/ur_robot_driver/resources/externalcontrol-1.0.5.urcap`.
Use the [official Humble robot setup guide](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_robot_driver/doc/installation/robot_setup.rst)
for the pendant procedure. The documented lab workflow starts the driver
first and then plays this program on the pendant. The current pendant
configuration and PolyScope version were not inspected during this review.

Extract and retain this particular arm's factory calibration when setting up
a new workstation. For example, choose a persistent configuration location:

```bash
mkdir -p "$UR5_VLA_ROOT/hardware_ws/config"
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=10.0.1.38 \
  target_filename:="$UR5_VLA_ROOT/hardware_ws/config/ur5_calibration.yaml"
```

Then supply `kinematics_params_file` with that file when launching the driver.
No arm-specific extracted calibration was found in the repository during
inspection; the Isaac `default_kinematics.yaml` is a simulation/default asset.
The installed MoveIt launch uses the description package's default kinematics,
so also configure a matching calibrated description for MoveIt if accurate
Cartesian geometry is required; passing the driver argument alone does not
update that separate launch.

### Bring up the physical arm and MoveIt Servo

Prepare every terminal using the environment/source commands above. Keep
long-running launch processes in separate terminals. The following controller
and teleoperation steps enable physical motion; use a clear swept path, low
pendant speed, and an accessible emergency stop.

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 \
  robot_ip:=10.0.1.38 \
  reverse_ip:=10.0.1.42 \
  launch_rviz:=false \
  initial_joint_controller:=forward_position_controller \
  activate_joint_controller:=false
```

If the extracted calibration exists, append
`kinematics_params_file:="$UR5_VLA_ROOT/hardware_ws/config/ur5_calibration.yaml"`.
Start the saved External Control program on the pendant. In a second terminal:

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 launch_rviz:=true launch_servo:=true
```

The installed MoveIt launch explicitly supports `launch_servo`. The driver
command above initially leaves the forward-position controller inactive.
Activate it for continuous Servo control, then start Servo:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
ros2 control list_controllers
ros2 topic echo /servo_node/status --once
ros2 topic echo /joint_states --once
```

Confirm `forward_position_controller` is active, the trajectory controller is
inactive, and Servo status is zero. The installed Servo configuration outputs
joint positions on `/forward_position_controller/commands` and uses a 0.1 s
incoming-command timeout. The command path is:

```text
Logitech → /joy → logitech_servo → /servo_node/delta_twist_cmds
  → MoveIt Servo → /forward_position_controller/commands
  → UR robot driver → physical UR5
```

If the pendant reports **Receive program failed: connection refused at
10.0.1.42:50002**, check `ss -ltnp | rg ':50002'`. That port is the driver's
External Control script server on the PC. A reachable dashboard at the robot
does not establish that this PC-side server is running. A crashed
`ur_ros2_control_node` can leave the dashboard and launch process alive while
port 50002 and joint feedback disappear.

Stop Servo and joystick command output, stop the old driver launch, then restore
the driver with motion controllers initially inactive:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 robot_ip:=10.0.1.38 reverse_ip:=10.0.1.42 \
  launch_rviz:=false initial_joint_controller:=forward_position_controller \
  activate_joint_controller:=false
```

After port 50002 and joint feedback return, press **Continue** on the pendant
to retry the failed program receive. Confirm the External Control connection
before reactivating the forward-position controller and Servo using the normal
teleoperation steps. Keep the joystick trigger released during reconnection.

### Cameras, joystick, and gripper

Connect both RealSense cameras over USB 3. Launch each camera in a separate
terminal using its serial number and the namespaces expected by the recorder:

```bash
# Base D435i
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_044322072365 camera_namespace:=base camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30

# Wrist D435
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_827312073590 camera_namespace:=wrist camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Verify RGB and aligned depth on both cameras:

```bash
ros2 topic hz /base/camera/color/image_raw
ros2 topic hz /wrist/camera/color/image_raw
ros2 topic hz /base/camera/aligned_depth_to_color/image_raw
ros2 topic hz /wrist/camera/aligned_depth_to_color/image_raw
```

Each `topic hz` command runs until interrupted. Enumerate the Logitech device,
replace the placeholder with its actual device ID, and start input:

```bash
ros2 run joy joy_enumerate_devices
ros2 run joy joy_node --ros-args \
  -p device_id:=YOUR_LOGITECH_DEVICE_ID \
  -p deadzone:=0.1 -p autorepeat_rate:=50.0
```

For a new-dataset episode, run the guarded reset **before starting the mapper**:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

This moves the arm to its standard pose, requests at most 20% speed scaling
when needed, checks an interpolated path with MoveIt, then restores Servo.
Wait for `DATASET START READY`, then start the mapper in another terminal:

```bash
ros2 run ur5_vla_dataset_recorder logitech_servo
```

| Logitech control | Behavior in the current mapper |
| --- | --- |
| Physical button 1 / trigger (`buttons[0]`) | Hold to enable arm Twist commands; release publishes zero Twist |
| Stick forward/back and left/right | Base-frame X/Y translation |
| Handle twist | Rotation about base-frame Z |
| Physical buttons 5 / 6 | Z down / up while trigger is held |
| Physical buttons 2 / 3 | Close / open Robotiq electric gripper through SetIO; these buttons do not require the arm trigger |
| Physical button 7, held for two seconds with trigger released | Guarded standard-pose reset, then resume teleoperation |
| Physical buttons 9 / 10 / 11 | Recorder start / discard / save-success when the recorder is running |

The mapper requests up to 0.30 m/s linear and 0.50 rad/s angular speed in
`base_link`; these are input command magnitudes, subject to Servo and robot
limits. Gripper digital outputs use pin 0 for close and pin 1 for open.
Operate the gripper once to initialize its estimated dataset state, then check
`/gripper/state` and `/gripper/command` with `ros2 topic echo ... --once`.
The convention is 0 = closed, 1 = open, with no measured finger/contact state.

### Record demonstrations and return to trajectory control

Start the 20 Hz recorder after the cameras, joint state, Servo status, arm
targets, and gripper streams are ready:

```bash
ros2 launch ur5_vla_dataset_recorder recorder.launch.py
ros2 param set /dataset_recorder task \
  "Pick up the orange cube and place it in the bowl"
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger "{}"
```

Set the task to describe the actual demonstration. Use button 9 to start,
button 11 to save success, or button 10 to discard. Raw recordings go to
`data/dataset/staging/episodes/`. Recording observes motion and does not
command the arm. The current recorder also requires aligned depth, fresh
inputs, Servo status zero, and at least 20 GB free disk space.

To return to MoveIt Plan and Execute, release the trigger and stop the mapper,
then stop Servo and switch controllers:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller
ros2 control list_controllers
```

### Daily run: teleoperation and dataset recording, one step at a time

Use this sequence after installation and workspace builds are complete. Each
long-running process has its own terminal. Prepare every terminal with step 1.
The reset and teleoperation commands physically move the arm: clear the swept
path, use low pendant speed, and keep the emergency stop accessible. Keep
standalone policy, replay, and `control_engine` processes stopped throughout.

**1. Prepare every new terminal.**

Use the workspaces inside this repository:

```text
physical-ur5-vla-implementation/
├── hardware_ws/install/   # This repository's hardware packages
└── ros2_ws/install/       # This repository's teleop and recorder package
```

Start from a fresh terminal. Source the system ROS installation first, then
the two repository overlays explicitly with `local_setup.bash`:

```bash
conda deactivate
export UR5_VLA_ROOT=/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/local_setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/local_setup.bash"
```

If Conda remains active, repeat `conda deactivate`. If Conda was not active,
continue with the source commands. ROS commands use system Python.

Verify the selected packages before starting hardware:

```bash
ros2 pkg prefix ur5_vla_dataset_recorder
ros2 pkg prefix robotiq_driver
ros2 pkg prefix ur_robot_driver
ros2 pkg prefix ur_moveit_config
/usr/bin/python3 -c 'import ur5_vla_dataset_recorder as p; print(p.__file__)'
```

The recorder prefix must be
`$UR5_VLA_ROOT/ros2_ws/install/ur5_vla_dataset_recorder`, and the Robotiq
prefix must be `$UR5_VLA_ROOT/hardware_ws/install/robotiq_driver`.
The recorder Python module must also resolve inside this repository; a
`ros2_ws/build/` path is normal for its symlink installation. The UR arm driver
and MoveIt are installed under `/opt/ros/humble`. These resolutions were
verified in a clean shell on 2026-10-13. If a package resolves from an older
workspace, stop and correct the terminal environment before proceeding.

**2. Check robot connectivity.**

```bash
ping -c 3 10.0.1.38
```

**3. Terminal 1 — start the physical UR5 driver.**

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 \
  robot_ip:=10.0.1.38 \
  reverse_ip:=10.0.1.42 \
  launch_rviz:=false \
  initial_joint_controller:=forward_position_controller \
  activate_joint_controller:=false
```

Apply the calibration argument described above if the extracted calibration
file is available. Then start the saved External Control program on the teach
pendant. Leave the driver running.

**4. Terminal 2 — start MoveIt and Servo.**

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 launch_rviz:=true launch_servo:=true
```

**5. Terminal 3 — activate the Servo controller and check readiness.**

Run these commands one by one:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
ros2 control list_controllers
ros2 topic echo /servo_node/status --once
ros2 topic echo /joint_states --once
```

Require the forward position controller active, the scaled trajectory
controller inactive, fresh joint state, and Servo status **0**.

**6. Terminal 4 — start the base D435i camera.**

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_044322072365 \
  camera_namespace:=base camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

**7. Terminal 5 — start the wrist D435 camera.**

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_827312073590 \
  camera_namespace:=wrist camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

**8. Terminal 3 — verify all camera streams.**

Run each command separately and press Ctrl+C after observing its rate:

```bash
ros2 topic hz /base/camera/color/image_raw
ros2 topic hz /wrist/camera/color/image_raw
ros2 topic hz /base/camera/aligned_depth_to_color/image_raw
ros2 topic hz /wrist/camera/aligned_depth_to_color/image_raw
```

**9. Terminal 6 — identify and start the Logitech joystick.**

```bash
ros2 run joy joy_enumerate_devices
```

Replace `YOUR_LOGITECH_DEVICE_ID` with the reported numeric device ID:

```bash
ros2 run joy joy_node --ros-args \
  -p device_id:=YOUR_LOGITECH_DEVICE_ID \
  -p deadzone:=0.1 -p autorepeat_rate:=50.0
```

**10. Terminal 3 — move to the recording start pose.**

Leave the teleoperation mapper stopped while this guarded reset runs:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

Wait for **`DATASET START READY`**. The command moves the arm and restores
Servo after the reset; a failed readiness check must be resolved first.

**11. Terminal 7 — start teleoperation.**

```bash
ros2 run ur5_vla_dataset_recorder logitech_servo
```

Hold physical button 1 (trigger) to move using the stick and handle twist.
Buttons 5/6 move down/up while the trigger is held. Release the trigger to
publish zero arm Twist. Buttons 2/3 close/open the gripper independently of the
trigger. Operate the gripper once to initialize its recording state. The
full mapping is in the control table above.

**12. Terminal 3 — check gripper streams.**

```bash
ros2 topic echo /gripper/state --once
ros2 topic echo /gripper/command --once
```

The values are 0 = closed and 1 = open; state is the acknowledged command
estimate rather than a measured finger position.

**13. Terminal 8 — start the dataset recorder.**

```bash
ros2 launch ur5_vla_dataset_recorder recorder.launch.py
```

**14. Terminal 3 — set the task to match the actual demonstration.**

```bash
ros2 param set /dataset_recorder task \
  "Pick up the orange cube and place it in the bowl"
```

**15. Check recorder readiness before starting an episode.**

```bash
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger "{}"
```

Resolve any missing/stale input or disk-space issue reported by the service.

**16. Record each demonstration with the joystick.**

1. Place the object and prepare the start pose.
2. Press physical **button 9** to start recording.
3. Hold the trigger and perform the task, using the gripper buttons as needed.
4. Release the trigger after the demonstration.
5. Press **button 11** to save a successful episode, or **button 10** to discard it.

Before the next episode, release the trigger and hold **button 7 for two
seconds** to run the guarded standard-pose reset through the mapper. Wait for
completion before starting another recording. Finish or discard the current
episode before resetting.

**17. Terminal 3 — inspect saved episodes.**

```bash
ls -lah "$UR5_VLA_ROOT/data/dataset/staging/episodes/"
```

Review the saved demonstrations before converting an explicit selection;
follow [Dataset conversion and validation](#dataset-conversion-and-validation).

**18. Finish teleoperation and return to trajectory control if needed.**

Release the trigger, finish or discard any active episode, and stop the mapper
in terminal 7 with Ctrl+C. In terminal 3:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller
ros2 control list_controllers
```

The controller switch returns the arm to MoveIt Plan and Execute. When ending
the session, stop the recorder, joystick input, cameras, MoveIt, and driver
with Ctrl+C in their respective terminals; stop the driver last. Follow the
lab procedure for ending the pendant program and powering down the robot.

### Setup limits and troubleshooting

If the driver cannot connect, check robot/PC Ethernet addressing, the URCap's
PC host address, and that the External Control program is playing. If the
arm does not respond to teleoperation, check `/joy`, controller activation,
the Servo start-service response, and `/servo_node/status`. If recording is
blocked, its status service reports readiness problems; check all four camera
streams and initialize the gripper estimate. An `_rclpy_pybind11` error usually
means a Conda Python environment was used instead of ROS's system Python.

The current MoveIt scene does not model the table, bowl, cameras, or people;
a successful collision query is not a check of the real swept path. The
installation, local launch arguments, and mapper configuration were inspected
for this chapter, but no physical bring-up or teleoperation was executed.
Supervised hardware validation remains required. Keep standalone
`scripts/control_engine*.py` and policy/replay publishers stopped during this
Servo workflow: they form a separate control path and can compete with it.

### Experiment 2: Sort the fruits demonstrations

Task label: **`Sort the fruits`**. Place two apples and two cubes in the
workspace. One complete episode picks both apples, one after another, and
places both in the bowl while leaving the cubes outside. Save success only
after the full two-apple sequence. These are manual demonstrations with the
regular joystick mapper; stop policy executors, standalone control engines
and any other joystick mapper first.

Successful episodes are saved separately in:

```text
data/dataset/staging/experiment_2_sort_fruits/
├── experiment.json
├── episodes/
│   └── episode_YYYYMMDD_HHMMSS/
│       ├── metadata.yaml
│       ├── frames.jsonl
│       ├── images/base/
│       ├── images/wrist/
│       ├── depth/base/
│       └── depth/wrist/
└── setup/ros_logs/
```

With cameras, driver, MoveIt/Servo and exactly one `joy_node` already running,
restore NORMAL safety mode and the running robot program on the pendant, then:

```bash
bash "$UR5_VLA_ROOT/scripts/record_experiment2_sort_fruits.sh"
```

The launcher checks command-source conflicts and robot run state, enables
continuous stationary Servo output (`moveit_servo.num_outgoing_halt_msgs_to_publish=0`),
starts Servo, then launches teleop and the recorder. Fresh hold targets allow
recording to start with the trigger released; Servo's command timeout remains
active. This parameter stays set for the current Servo process.
It does not move to a start pose
or clear physical safety stops. To prepare only the nodes while Servo remains
stopped, use `--prepare-only`. After manual robot recovery, keep the trigger
released and start Servo separately:

```bash
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
```

Press physical **button 3** to open/initialize the gripper before recording.
Check readiness:

```bash
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger "{}"
```

| Physical joystick button | Action |
| --- | --- |
| 9 | Start one full sorting episode |
| 10 | Stop and discard/abort the episode |
| 11 | Stop and save the successful two-apple episode |
| Trigger / 1 | Hold for arm motion; release to command zero Twist |
| 2 / 3 | Close / open gripper |
| 5 / 6 with trigger | Move down / up |

Recording is configured at 20 Hz with both RGB cameras, both aligned depth
streams, joint positions, arm targets, Servo status and gripper command/state
estimates. Configuration:
`ros2_ws/src/ur5-vla-dataset-recorder/config/recorder_experiment2_sort_fruits.yaml`.
The launcher also starts `scripts/demonstration_monitor.py`, a read-only evidence
observer following confirmed recorder state (polled every 0.25 seconds).
Each episode receives a separate record under
`data/dataset/staging/experiment_2_sort_fruits/telemetry/episode_YYYYMMDD_HHMMSS/`:
`observations.jsonl`, `summary.json`, `analysis.txt`, camera videos and snapshots,
sampled depth, configuration and available TCP poses. Native joint positions,
velocities/efforts when published, arm targets, joystick, gripper IO, robot/Servo
status and ROS warnings/errors are logged with receive timestamps. A shared
`telemetry/episodes.csv` indexes completed records. Button 11 supplies the operator
success label; button 10 retains aborted evidence while discarding the training
episode. Actual contact, apple identity and individual placement times are not
automatically established. Telemetry may start slightly after dataset capture;
the original 20 Hz dataset retains its own timestamps. Launching the observer
mid-episode cannot recover earlier live telemetry.
The `episode_subdirectory` parameter routes this experiment to its own folder;
the default remains `staging/episodes` for existing workflows. Aborted episodes
are discarded, while successful saves finalize the incomplete directory
transactionally. Reset the scene between episodes and do not label an episode
successful after placing only one apple. Conversion later uses
`data/dataset/staging/experiment_2_sort_fruits/episodes` as its explicit input
root and a new output name under `data/dataset/lerobot`.

## Directory map

| Path from repository root | Contents |
| --- | --- |
| `hardware_ws/src/` | Robotiq controller, driver, description, serial library, and UR5 Isaac simulation source |
| `ros2_ws/src/ur5-vla-dataset-recorder/` | Current ROS package, launch/config files, tests, tools, and package notebooks |
| `scripts/` | Standalone control, replay, and offline/shadow test entrypoints |
| `notebooks/` | Four later SmolVLA and π0.5 Colab notebooks and troubleshooting copies |
| `docs/episode_audit_20261002/` | Reviewed episode lists, audit code, contact sheets, and reports |
| `docs/thesis_chapter/` | Canonical methodology chapter, builder, figures, PDF, DOCX, and evidence register |
| `data/dataset/` | Raw episodes (`staging/`), converted datasets (`lerobot/`), and quarantine |
| `data/models/` | Trained checkpoints and evaluation results |
| `data/huggingface_cache/` | Copied model/dataset Hub cache used by fresh terminals |
| `data/hf_lerobot_cache/` | Copied LeRobot local dataset cache used by fresh terminals |
| `dump/desktop_vla/` | Earlier experiments, source code, simulation assets, and outputs |
| `dump/vla_script_training/` | Earlier LeRobot checkout, conversion/training scripts, and outputs |
| `dump/downloads_physical_ur5/` | Earlier downloaded training/deployment materials |
| `dump/downloaded_models/` | Downloaded checkpoints, including the 20 GB `Checkpoints/` archive |

See [Workspaces and hardware](#workspaces-and-hardware) for the role of each
hardware workspace package.

To regenerate the canonical chapter, run `/usr/bin/python3
"$UR5_VLA_ROOT/docs/thesis_chapter/build_chapter.py"`. Graphviz and system
matplotlib are required. Markdown and python-docx are installed locally under
`.python_deps/` (excluded from Git); on a new machine, install them with
`/usr/bin/python3 -m pip install --target "$UR5_VLA_ROOT/.python_deps"
markdown python-docx`.

The original project directories were copied, not moved. They are no longer sourced by the user's shell startup. Historical logs, report JSON, notebook output cells, and the dated handoff document may still quote original paths as evidence of where an experiment ran; they are not current runtime configuration.

## Current data and model

The `data/` layout below was verified on 2026-10-11. Paths are relative to
the repository root; episode media and checkpoint internals are omitted from
the tree for readability.

```text
data/
├── dataset/
│   ├── staging/
│   │   ├── episodes/                         # 112 finalized raw episodes
│   │   └── clean_20_cube_raw_nonconverted/   # Separate raw selection
│   ├── quarantine/
│   │   ├── invalid_episodes_2026-08-25/
│   │   └── partial_conversions/
│   └── lerobot/
│       ├── ur5_cube_110ep_v3/
│       ├── ur5_cube_30ep/
│       ├── ur5_cube_clean_20ep/
│       ├── ur5_cube_clean_20ep_delta/
│       ├── ur5_cube_clean_53ep/
│       ├── ur5_cube_clean_73ep_v2/
│       └── ur5_orange_cube_v1/
├── models/
│   ├── checkpoints/
│   │   └── ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/
│   └── evaluation/                         # Empty after cleanup
├── hf_lerobot_cache/
│   ├── lerobot/
│   └── local/
└── huggingface_cache/
    ├── datasets/
    └── hub/
```

| Folder | Purpose and contents |
| --- | --- |
| `data/dataset/staging/` | Original demonstrations before conversion. Episodes contain `frames.jsonl`, `metadata.yaml`, base/wrist RGB images, and depth when recorded. Active or unfinished recorder episodes use hidden `.incomplete` directories. The separate `clean_20_cube_raw_nonconverted/` selection is retained. |
| `data/dataset/quarantine/` | Preserved invalid recordings and interrupted conversion outputs, kept separate from the selected training datasets. |
| `data/dataset/lerobot/` | Converted, versioned training datasets. Each dataset keeps its metadata, frame data, and camera media together. Absolute and delta action datasets are distinct versions. |
| `data/models/checkpoints/` | Trained policy weights, model/training configuration, preprocessing and postprocessing assets, intermediate checkpoints, training logs, metrics, environment records, and checksum records. |
| `data/models/evaluation/` | Destination for future prediction CSVs, evaluation reports, and test logs. All previous contents were deleted on 2026-10-11; the directory remains empty. |
| `data/hf_lerobot_cache/` | LeRobot local dataset cache selected by `HF_LEROBOT_HOME`. |
| `data/huggingface_cache/` | Hugging Face model and dataset download caches selected by `HF_HOME`. |

`data/dataset/` contains only `staging/`, `quarantine/`, and `lerobot/`.
`data/models/` contains only `checkpoints/` and `evaluation/`. The previous
`data/ur5_vla_dataset/` name has been replaced, and its `exports/` folder was
deleted. Dataset and checkpoint contents were preserved during the directory
reorganization; evaluation contents were intentionally removed. The recorder
and model-check script use the new paths.

`data/dataset/staging/episodes/` contains 112 finalized raw episode directories. `data/dataset/lerobot/ur5_cube_110ep_v3/` is the reviewed 110-episode dataset: **285,204 frames at 20 Hz**. Its `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `data/`, and `videos/` form the LeRobot dataset; keep the folder intact. Other `lerobot/` children are earlier dataset versions and must be identified by exact folder name when reporting results.

The final 18,000-step SmolVLA checkpoint is:

```text
data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model/
```

It contains `model.safetensors`, `config.json`, the training config, and policy pre/postprocessor files. `data/models/evaluation/` is currently empty after the requested cleanup; future model checks write their reports there. `data/dataset/quarantine/` contains rejected or interrupted material; `data/dataset/staging/` contains raw recording material. The former `exports/` folder was deleted. Do not silently mix those categories when training.

Converted dataset folders currently include `ur5_cube_30ep`,
`ur5_cube_clean_20ep`, `ur5_cube_clean_20ep_delta`,
`ur5_cube_clean_53ep`, `ur5_cube_clean_73ep_v2`,
`ur5_orange_cube_v1`, and `ur5_cube_110ep_v3`. The folder name identifies
the selection and conversion; check its `meta/info.json` and
`meta/ur5_vla_conversion.json` before comparing experiments. Raw episode
directories contain `frames.jsonl`, metadata, and referenced camera media;
the README describes them by schema and dataset version rather than listing
all 1.4 million files individually.

At the original migration, the main raw/converted data copy completed with **1,427,363 files** in both source and destination trees and an rsync dry run showing zero differences. The copied hardware source, later checkpoint archive, LeRobot cache, and Hugging Face Hub/datasets caches also showed zero differences. Migration-time size estimates were about 139 GB of project data, roughly another 32 GB of copied caches, and more than 110 GB of older experiments/checkpoints; these are historical estimates before the exports/evaluation cleanup. These large folders are ignored by Git: **a normal Git push does not upload datasets or model weights**.

## Dataset conversion and validation

### Scripts, environment, and data flow

Conversion runs locally after recording and reviewing demonstrations:

```text
data/dataset/staging/episodes/<selected episodes>
  → raw validation and full media decoding
  → convert_lerobot.py (LeRobot dataset creation)
  → data/dataset/lerobot/<new dataset name>
  → validate_lerobot.py (converted dataset checks)
  → training notebook for the chosen policy
```

| File relative to the repository root | Role |
| --- | --- |
| `ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/convert_lerobot.py` | Main raw-to-LeRobot converter; invoked as a Python module |
| `ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/raw_dataset.py` | Raw metadata, timing, state/action, and media validation helpers |
| `ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/validate_lerobot.py` | Loads the converted dataset using LeRobot and writes a validation report |
| `docs/episode_audit_20261002/convert_reviewed_episodes.sh` | Wrapper for the dated 37-episode and combined 110-episode selections; runs conversion and validation |
| `docs/episode_audit_20261002/accepted_new_episodes.txt`, `combined_episodes.txt` | Exact episode selections used by that wrapper |

**`pi0` is the name of the local Conda environment**, not the model chosen by
the converter. It has LeRobot 0.5.2 and the Python dependencies needed for
conversion, video encoding, and validation. Its editable LeRobot installation
points to `dump/vla_script_training/lerobot`. The resulting dataset can be used
with SmolVLA or another compatible policy. ROS nodes use system Python;
conversion uses this separate environment and does not command hardware.

Prepare the source path and verify the installed environment:

```bash
export UR5_PACKAGE="$UR5_VLA_ROOT/ros2_ws/src/ur5-vla-dataset-recorder"
export PYTHONPATH="$UR5_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
cd "$UR5_PACKAGE"

/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -c 'import sys, lerobot; import importlib.metadata as m; print(sys.executable); print(lerobot.__file__); print(m.version("lerobot"))'
```

Set `UR5_VLA_ROOT` as described in [Location and environment](#location-and-environment)
before these commands. The absolute Conda executable avoids depending on shell
activation. The converter source still writes the historical compatibility
label `0.4.3 / dataset codebase v3.0` into conversion metadata; this label is
distinct from the installed LeRobot version checked above.

### Convert a new reviewed selection

Choose a **new dataset name**, set the task to match the demonstration, and
replace both example episode names with existing reviewed directory names.
Use the same terminal prepared above:

```bash
UR5_DATASET_NAME=MY_NEW_DATASET
UR5_TASK="Pick up the orange cube and place it in the bowl"
UR5_EPISODES=(episode_YYYYMMDD_HHMMSS episode_YYYYMMDD_HHMMSS)

/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.convert_lerobot \
  --input-root "$UR5_VLA_ROOT/data/dataset/staging/episodes" \
  --output-root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --task "$UR5_TASK" \
  --episodes "${UR5_EPISODES[@]}" \
  --depth none --storage video --action-mode absolute
```

For a longer reviewed selection, set the same array from your own text file
containing one episode directory name per line:

```bash
mapfile -t UR5_EPISODES < /absolute/path/to/reviewed_episode_names.txt
```

Load that list before running the conversion command. Explicit selection
keeps rejected recordings out of the dataset. Omitting `--episodes` selects
all finalized episode directories under the input root; the current raw
folder includes recordings excluded from the reviewed 110-episode dataset,
so omission is not equivalent to the reviewed selection.

The converter validates **all selected episodes before creating the output**.
It requires successful finalized recordings and checks metadata, canonical
joint ordering, frame/timestamp consistency, finite state/action values,
gripper conventions, and referenced media. It decodes all referenced RGB and
depth media during preflight, even when depth is excluded from the output.
Any raw validation failure stops conversion. Existing output directories are
refused by default. `--overwrite` deletes an existing output directory, so use
a new version name for routine conversions.

### Conversion options and resulting schema

| Option | Meaning |
| --- | --- |
| `--input-root` | Directory containing finalized raw episode folders |
| `--output-root` | New converted dataset directory |
| `--repo-id` | LeRobot dataset identifier, such as `local/MY_NEW_DATASET`; this command does not upload it to the Hub |
| `--episodes` | Explicit episode directory names; omission selects all finalized episodes |
| `--task` | Override the task for every selected episode; omission uses each episode's metadata task |
| `--depth none` | Current RGB-only training workflow; source depth files remain intact |
| `--depth meters` | Add float32 depth in metres and validity masks; requires suitable raw depth and metadata |
| `--storage video` | Store base/wrist RGB as videos; `image` is the alternative |
| `--action-mode absolute` | Six absolute joint-position targets in radians and an absolute binary gripper command |
| `--action-mode delta` | Six same-frame differences `q_command - q_measured`; gripper remains absolute |
| `--overwrite` | Replace an existing output by deleting it; not used in the commands above |

The current schema expects 640×480 camera images and creates a 20 Hz dataset.
`observation.state` contains six measured joints in canonical UR5 order plus
the estimated gripper state. `action` contains the six arm targets or deltas
plus the gripper command. The gripper convention is **0 = closed, 1 = open**.
Exact raw episode timing is retained as `source_episode_time_s` alongside
LeRobot's regular frame timestamps. The task is stored with each frame.
Raw recordings are read without being changed.

The output contains `meta/`, frame data under `data/`, and camera media under
`videos/` for video storage. `meta/ur5_vla_conversion.json` records the source
episode names, joint order, units, action semantics, gripper convention, depth
mode, and storage choice. Keep the whole dataset directory together.

### Validate the converted dataset

After conversion succeeds, run the validator in the same prepared terminal:

```bash
/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.validate_lerobot \
  --root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --expected-episodes "${#UR5_EPISODES[@]}" \
  --task "$UR5_TASK" \
  --video-backend pyav \
  --report "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME/validation_report.md"
```

Add `--expected-frames NUMBER` when the reviewed frame total is known. The
validator scans every state/action row, checks seven-dimensional finite values
and binary gripper actions, checks conversion semantics and optional expected
counts/task, and requires close/open transitions in each episode. Delta
datasets are also checked against a 0.012566 rad arm-delta envelope. It decodes
both camera views at the beginning, middle, and end of each episode; converted
video validation is sampled, rather than a full-frame decode. A passing run
prints `PASS`; validation errors produce a nonzero exit and a report.

### Existing reviewed 37- and 110-episode workflows

The dated wrapper uses fixed selections and destinations:

| Mode | Selection | Output dataset | Expected frames |
| --- | --- | --- | --- |
| `new` | 37 accepted October 1 episodes | `ur5_cube_oct01_37ep` | 88,497 |
| `combined` | Previous 73 plus the accepted 37 | `ur5_cube_110ep_v3` | 285,204 |

```bash
# Show the wrapper's available modes.
bash "$UR5_VLA_ROOT/docs/episode_audit_20261002/convert_reviewed_episodes.sh" --help

# Convert and validate the reviewed 110-episode selection.
bash "$UR5_VLA_ROOT/docs/episode_audit_20261002/convert_reviewed_episodes.sh" combined
```

Use `new` instead of `combined` for the fixed 37-episode selection. The
110-episode output **already exists**, so `combined` currently refuses to
overwrite it. These modes are historical selections, not automatic selection
of newly recorded episodes. For another selection, use the explicit new-name
command above. The combined selection retains a historical episode with
multiple grasp/release transitions, as documented in the audit report.

### Progress and troubleshooting

`[preflight]` messages show raw validation and media decoding; `[convert]`
messages show frame staging and video encoding; `[finalize]` marks metadata
and statistics completion. Encoding can take time. A failed or interrupted
conversion may leave a partial output directory: inspect it and keep it
separate from valid datasets before retrying with a new output name.

If imports fail, verify the `pi0` environment and `PYTHONPATH` above. If the
output already exists, choose a new dataset name. If preflight fails, inspect
the reported episode/media/timing error and revise the reviewed selection.
If the converted validator fails, retain its report for diagnosis and do not
use that output as a validated training dataset. Local conversion does not
need the robot running. Package conversion notebooks provide a separate Colab
route using cloud paths; archived `dump/` CSV converters use an older raw
format and do not replace this `frames.jsonl` workflow.

## ROS package files

The installed console commands are defined in `ros2_ws/src/ur5-vla-dataset-recorder/setup.py`. These modules live in its `ur5_vla_dataset_recorder/` directory:

| Module | Purpose |
| --- | --- |
| `recorder_node.py`, `episode_writer.py` | 20 Hz synchronized recording and file writing; default output is this repository's data folder |
| `logitech_servo_node.py`, `standard_pose_node.py` | Joystick teleoperation, gripper IO, and guarded standard-pose preparation |
| `core.py`, `raw_dataset.py` | Joint ordering, schema checks, and raw episode validation |
| `convert_lerobot.py`, `validate_lerobot.py` | Raw-to-LeRobot conversion and converted-dataset validation |
| `action_semantics.py`, `episode_control.py`, `episode_execution.py` | Absolute/delta action conventions and episode state/control helpers |
| `safety.py`, `live_control.py` | Readiness gates and live-control checks |
| `policy_executor.py`, `shadow_protocol.py` | Read-only live shadow client and localhost prediction protocol |
| `policy_inference_server.py`, `offline_inference.py` | Model server and saved-dataset inference; offline inference does not move the robot |
| `policy_live_executor.py`, `policy_episode_executor.py` | Live policy execution and bounded episode execution; these can command hardware |
| `package_dataset.py`, `split_archive.py`, `install_checkpoint.py` | Dataset archive and checkpoint transport utilities |

`launch/recorder.launch.py` loads `config/recorder.yaml`. The recorder config uses `data/dataset` relative to this repository, and the recorder resolves relative `output_root` values from the repository root. `test/` contains package checks. `tools/` contains notebook generators for the 30-, 20-, 53-, 73-, and 110-episode experiments, generic conversion-notebook generators, and a historical training progress monitor. That monitor uses the copied `dump/downloaded_models/Checkpoints/` path and an old process ID; it is not a live monitor for a new run.

The package's `notebooks/` contains conversion notebooks and SmolVLA/π0/π0.5 training notebooks for named dataset versions. Colab notebooks intentionally use `/content` and Google Drive paths because they execute in Colab. `notebooks/` at repository root contains four later 110-episode training and π0.5 troubleshooting notebooks. All 17 package/root notebook files parsed as JSON; cloud training was not rerun during this migration.

Package conversion notebooks: `convert_raw_ur5_to_lerobot_colab.ipynb`,
`convert_ur5_cube_30ep_colab.ipynb`, and
`convert_ur5_cube_clean_20ep_colab.ipynb`. Package SmolVLA notebooks:
`train_ur5_smolvla_colab.ipynb`, `train_ur5_cube_30ep_smolvla_40000_colab.ipynb`,
the 20- and 53-episode clean variants, the 73-episode A100 variant, and
`train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb`. The package also has
`train_ur5_cube_clean_73ep_v2_pi0_a100_colab.ipynb` and
`train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb`. The root notebook files are
`train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb`,
`train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb`,
`pi05_error_20261004.ipynb`, and `pi05_fixed_auth_20261004.ipynb`.

Notebook generators in `tools/` are `generate_colab_notebook.py`,
`generate_conversion_colab_notebook.py`, `generate_generic_conversion_colab.py`,
`generate_30ep_training_colab.py`, `generate_clean_20ep_colab.py`,
`generate_clean_20ep_b8_60k_guarded_colab.py`,
`generate_clean_53ep_80k_colab.py`, `generate_clean_53ep_b8_60k_colab.py`,
`generate_pi0_73ep_a100_colab.py`, `generate_110ep_training_colab.py`, and
`generate_pi05_110ep_colab.py`. They regenerate specific notebook variants;
review the named input/output notebook before rerunning a generator.

`docs/episode_audit_20261002/` contains `audit_episodes.py` (raw validation
and contact sheets), `finalize_report.py` (selection and report),
`convert_reviewed_episodes.sh` (new/combined conversion),
`accepted_new_episodes.txt` and `combined_episodes.txt` (exact selections),
CSV/JSON audit summaries, contact sheets, and `REPORT.md`. The canonical
`docs/thesis_chapter/` contains `build_chapter.py`, the Markdown chapter,
generated HTML/DOCX/PDF, figures, checkpoint comparison CSV, and a SHA-256
evidence register. `docs/SMOLVLA_110EP_COLAB_README.md` is the training
notebook handoff note; this root README remains the current path guide.

## Local experiment 1 Pi0 training

`scripts/train_experiment1_pi0.sh` trains the Pi0 action expert and projections
on `data/dataset/lerobot/ur5_cube_110ep_v3` (110 episodes, 285,204 frames).
The vision-language backbone stays frozen. This is Pi0 expert finetuning,
distinct from the existing SmolVLA checkpoint and from Pi0.5.

The local profile uses the three GPUs numbered 0–2, float32 master weights with
FP16 mixed precision and gradient scaling, FSDP sharding,
gradient checkpointing, one example per GPU and four accumulation steps
(effective batch 12). GPU 3 remains available for the desktop. CPU affinity,
low process/I/O priority and zero data-loader workers reduce competition with
experiment-2 recording. Recording readiness and dropped samples still need
monitoring; shared hardware cannot guarantee zero interference.

```bash
# Verify loading and two optimizer steps without creating a W&B run.
bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh" --probe

# Start a 20,000-step run with online W&B graphs.
bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh"

# Optional slower numerical reference profile:
UR5_PI0_PRECISION=float32 bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh" --probe
```

Use one training launcher at a time. Outputs are isolated under
`data/models/checkpoints/experiment1_pi0_local/`, with a unique run directory.
`terminal.log` records progress; `metrics.jsonl` stores measured training metrics;
`wandb_run.json` contains the dashboard URL. W&B authentication must already be
configured in the `pi0` environment. Training graphs describe optimization loss,
not physical task success; this profile has no held-out evaluation split.
Only the latest two complete checkpoints are retained to reserve recording
space. Each includes model configuration, weights, processors and optimizer
state. The training process does not publish robot commands or replace the
deployed SmolVLA model.

### Automatic local Pi0 checks and checkpoint recovery

The workstation's enabled user timer `ur5-pi0-training-watchdog.timer` checks
the designated job at 00:00, 02:00, 04:00, etc. in `America/Toronto`. It operates
independently of this chat; user lingering is enabled so logout does not stop
the timer. The workstation and dataset drive must remain available.

```bash
# Install/reinstall the timer and perform an immediate health check.
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/install_pi0_training_watchdog.py"

# Inspect the schedule and check history.
systemctl --user list-timers ur5-pi0-training-watchdog.timer
journalctl --user -u ur5-pi0-training-watchdog.service -n 20 --no-pager

# Manual checkpoint recovery, after the old training process has stopped.
bash "$UR5_VLA_ROOT/scripts/train_experiment1_pi0.sh" --resume \
  "$UR5_VLA_ROOT/data/models/checkpoints/experiment1_pi0_local/ur5_cube_110ep_pi0_expert_20261019_210719"

# Disable automatic recovery before deliberately stopping training.
systemctl --user disable --now ur5-pi0-training-watchdog.timer
```

`scripts/pi0_training_watchdog.py` verifies exact training command arguments and
recent metric updates. It resumes a stopped job, or a job with no new metrics
for 30 minutes, from the latest complete numbered checkpoint. Recovery runs in
its own user service, `ur5-pi0-auto-recovery.service`. A global training lock
prevents duplicate launches. Checks and recovery decisions are saved in
`data/models/checkpoints/experiment1_pi0_local/watchdog_checks.jsonl` and
`watchdog_status.json`. Three recovery attempts within 24 hours trigger a logged
block for inspection; low disk space also blocks restart.

Recovery restores weights, Adam optimizer moments, FP16 scaler, the original
step budget, learning-rate schedule and deterministic sampler position. It
continues the same W&B run. Future checkpoints also retain per-rank Torch RNG
state. Original checkpoints did not save RNG state, so their recovery restores
the training state but not the exact original random-noise sequence. Unsaved
updates may repeat: the currently running original process saves every 1,000
steps; recovery launches use 150-step checkpoint intervals. Only complete
checkpoint directories are selected, and the latest two are retained.

### Experiment 1 Pi0 on Google Colab A100

Open the [uploaded Pi0 110-episode Colab notebook](https://colab.research.google.com/drive/1fWyodT8BOWv1MfMSBM3uCS7h1PMvnfD3).
The local copy is `notebooks/train_ur5_cube_110ep_v3_pi0_a100_colab.ipynb`;
regenerate it with
`/usr/bin/python3 ros2_ws/src/ur5-vla-dataset-recorder/tools/generate_pi0_110ep_colab.py`.

Select **Runtime → Change runtime type → A100 GPU**, add `HF_TOKEN` and
`WANDB_API_KEY` to Colab Secrets with notebook access, then **Runtime → Run all**.
Authorize Drive when prompted. The existing source folder is
`MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5/datasets/lerobot/ur5_cube_110ep_v3`;
no new dataset upload is needed. A fresh Colab runtime is required for its pinned
LeRobot 0.6.1 software stack. The notebook itself has not been GPU-executed locally.

This starts a fresh Pi0 expert fine-tune from `lerobot/pi0_base`, using bfloat16,
absolute actions and mean/std normalization. It validates the dataset, benchmarks
actual optimizer steps, chooses a measured batch size, and trains for 20,000 steps.
W&B logs online to `ur5-experiment1-pi0`; the URL is printed and saved in the Drive
run folder. Every 2,500 steps a complete checkpoint is checksum-copied into
`pHYSICAL UR5/checkpoints/ur5_pi0_trained/<run-name>/checkpoints/`. Reports and a
final portable checkpoint tar/checksum are also exported. Completed Drive
checkpoints persist across disconnects; the notebook contains manual resume
instructions. A fresh Run all starts a new run rather than resuming automatically.
The local FSDP job is a separate backup; move execution only after confirming
the cloud job is actually training.

## Standalone scripts

Some filenames resemble ROS package modules, but they are distinct programs,
not duplicate copies. The files under `ros2_ws/src/ur5-vla-dataset-recorder/`
are installed by `colcon` and invoked with `ros2 run`; they share tested
package helpers and accept explicit artifact paths. In particular,
`ros2 run ur5_vla_dataset_recorder policy_executor` is a **shadow-only**
inference client and rejects `--live`. The package's
`policy_live_executor` and `policy_episode_executor` are separate guarded
programs; the latter defaults to shadow mode.

The root `scripts/` folder preserves standalone deployment and experiment
entrypoints from the earlier `bypass` workflow. `scripts/policy_executor.py`
selects the local 110-episode checkpoint, starts the inference server, and
publishes seven abstract actions on `/control_abstraction_layer` for the
separate `control_engine.py` to consume. It imports observation/protocol
helpers from the ROS package; it is not the same code as the package's
shadow-only `policy_executor.py`. `scripts/policy_executor_phased.py` is an
earlier phase-aware variant. `scripts/episode_executor.py` replays saved
actions to the abstraction topic, whereas the package's
`policy_episode_executor.py` executes new policy predictions with explicit
shadow/physical modes. Keep these roles distinct when choosing a command;
the standalone scripts can cause physical motion when paired with the live
control engine.

| File in `scripts/` | Purpose and status |
| --- | --- |
| `project_paths.py` | Derives the repository, package, data, downloaded-model, and cache paths; gives model-server subprocesses a clean project environment |
| `execution_profile.py` | Shared standalone speed defaults and validation: 0.5 speed scale, 10 executed actions per prediction |
| `control_engine.py` | Smooths abstract commands and publishes arm/gripper commands; **physical motion** unless in dry-run mode |
| `control_engine_phased.py` | Phase-aware control-engine variant; **physical motion** unless in dry-run mode |
| `policy_executor.py` | Starts the copied 110-episode SmolVLA server and publishes 7D actions; **physical motion** |
| `trajectory_offset.py` | Imported helper for continuous Cartesian translation of policy arm targets; uses the MoveIt robot description, local kinematics, and MoveIt path checks |
| `grasp_offset.py` | Shared offset validation/planning and the earlier one-grasp intervention; imported by the executor, not run separately |
| `policy_executor_phased.py` | Phase-gated executor using the 110-episode checkpoint; **physical motion** outside shadow mode |
| `episode_executor.py` | Replays saved episode actions to the abstraction topic; **physical motion** |
| `standard_pose.py` | Standalone standard-pose helper that publishes controller targets; **physical motion** |
| `test_110ep_model.sh` | `offline`, `server`, and read-only `shadow` model checks using this repository's paths |
| `pi05_110ep_dataset_validation.json`, `smolvla_110ep_notebook_validation.json` | Saved validation records, not executables |
| `thesis_chapter/` | Duplicate snapshot from initial copy; use `docs/thesis_chapter/` as canonical |

The copied live scripts import the ROS package from this repository's source tree before another installation can be chosen. Their model and data constants resolve under this repository. The original standalone scripts outside this repository were not edited.

`standard_pose.py` exits automatically after `Standard pose command completed`.
It cancels its timer and leaves the spin loop before shutting ROS down; shutdown
inside the timer callback previously hung the global executor. Ctrl+C also exits
cleanly. Completion means the final target was published, not independently
verified physical arrival. This helper does not clear control-engine faults;
after inspecting a previous execution failure, stop the old executor and use
`/control_engine/reset` before preparing another policy episode.

## Policy execution: setup, commands, offsets, and diagnostics

This chapter covers the current standalone `scripts/policy_executor.py` and
`scripts/control_engine.py` workflow. These scripts can command the physical
UR5. The installed ROS command `ros2 run ur5_vla_dataset_recorder
policy_executor` is a different, shadow-only program. Choose the workflow
explicitly using the commands below.

### Components, checkpoint, and action meaning

```text
Base/wrist RGB + measured UR5 joints + estimated gripper state + task
  → scripts/policy_executor.py (system Python / ROS)
  → localhost:8765 model server (pi0 environment / CUDA)
  → checkpoint preprocess → SmolVLA → checkpoint postprocess
  → optional Cartesian offset of arm targets
  → /control_abstraction_layer (six absolute joints + binary gripper)
  → scripts/control_engine.py
  → /forward_position_controller/commands + UR SetIO
  → physical UR5 and Robotiq electric gripper
```

| Setting | Current value |
| --- | --- |
| Checkpoint | `data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model/` |
| Model | SmolVLA fine-tuned on the 110-episode UR5 dataset for 18,000 steps |
| Task constant | `Pick up the orange cube and place it in the bowl` |
| Policy observations | Base/wrist 640×480 RGB, six measured joints in canonical UR5 order, estimated gripper state |
| Policy output | 50 actions per prediction, each containing six absolute joint positions in radians and one gripper value |
| Policy publication | Default 10 Hz (`--speed-scale 0.5`); executes first 10 of 50 actions, spanning 1 second before replanning |
| Gripper convention | Threshold at 0.5; 0 = closed, 1 = open |
| Control-engine publication | Nominal 125 Hz, with joint filtering and velocity/acceleration limits |
| Model runtime | `pi0` Conda environment; the ROS executor/engine use `/usr/bin/python3` |
| Model server | `127.0.0.1:8765`, started and terminated by the standalone executor |
| Offset helpers | `trajectory_offset.py` and `grasp_offset.py` are imported helpers; do not run them separately |

Both standalone policy executors select the checkpoint above through
`MODEL_PATH` in their source. There is currently no `--checkpoint` argument on
`scripts/policy_executor.py`; change that source constant for another compatible
checkpoint. Use `--task` to change its language instruction. The control engine has no model
or dataset path. `project_paths.py` resolves paths from this repository root.
The model server loads the saved preprocessing/postprocessing assets alongside
weights; retain the complete `pretrained_model/` directory.

### 1. Prepare every terminal and check paths

Use a fresh terminal for each long-running process:

```bash
conda deactivate
export UR5_VLA_ROOT=/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/local_setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/local_setup.bash"
```

Repeat `conda deactivate` if another Conda environment remains active. Verify
that the repository overlays are selected and the GPU is available:

```bash
ros2 pkg prefix ur5_vla_dataset_recorder
ros2 pkg prefix robotiq_driver
nvidia-smi
ls -lh "$UR5_VLA_ROOT/data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model/model.safetensors"
```

The first two prefixes must be inside this repository's `ros2_ws/install/`
and `hardware_ws/install/`. System ROS packages such as `ur_robot_driver`
and `ur_moveit_config` remain under `/opt/ros/humble`.

### 2. Check the checkpoint without physical policy output

Keep standalone control engines and physical policy/replay executors stopped
for these checks. Saved-observation inference requires no running robot:

```bash
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" offline
```

This checks three samples per episode (330 observations for the 110-episode
dataset), writes CSV/JSON reports to `data/models/evaluation/smolvla_110ep_18000/`,
and returns a failure if the configured prediction envelope is exceeded. The
wrapper's offline/server modes require a working NVIDIA GPU.

To inspect predictions from live observations without command publishers,
start the normal driver, cameras, and Servo observation stack described in
[UR5 physical arm installation and teleoperation](#ur5-physical-arm-installation-and-teleoperation),
then run these in separate terminals:

```bash
# Model-server terminal, for the read-only shadow workflow only.
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" server
```

```bash
# ROS shadow-client terminal; does not publish arm or gripper commands.
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" shadow
```

Stop both with Ctrl+C before starting the standalone physical executor. Its
own server also uses port 8765. Do not run both servers simultaneously. The
wrapper fixes inference seed 42; the standalone live executor does not set an
inference seed, so sampled predictions can differ between these workflows.

### 3. Prepare the physical robot before starting the engine

Follow the physical-arm chapter to start the UR driver, play External Control
on the pendant, start MoveIt, and start both RGB cameras. Reuse its camera
commands and serial numbers. Activate `forward_position_controller`, use the
same trained task/object/camera setup, and keep the pendant speed low (the
lab trials used 20%). Inspect the speed feedback:

```bash
ros2 control list_controllers
ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once
```

With policy executors, replay scripts, and control engines stopped, prepare
the guarded starting pose:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

Wait for `DATASET START READY`. Initialize/open the Robotiq electric gripper through
the established teleop workflow if needed, then stop `logitech_servo` with
Ctrl+C before policy operation. The default policy gripper estimate is open,
so begin with the physical gripper open and no held object.

Stop Servo command production before starting the standalone engine:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"
```

Keep MoveIt running: offset planning uses `/move_group/get_parameters`,
`/compute_fk`, `/compute_ik` for grasp mode, and `/check_state_validity`.
The standalone engine writes joint targets directly to the forward position
controller and waits for a successful Servo-stop acknowledgment before accepting
policy commands. This differs from the
teleoperation path through Servo. Do not mix their command sources.

### 4. Control-engine terminal — start arm/gripper command handling

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 control_engine.py --speed-scale 0.5
```

The engine starts in live mode with a default speed scale of 0.5. It waits for measured joints and abstract
policy commands, smooths arm targets, and sends two sequential SetIO writes
for each accepted gripper change. After both writes succeed it publishes
acknowledged command history on `/control_engine/gripper_state` at 20 Hz;
this is not measured finger position or grasp success.

Gripper control is automatic during policy execution. The engine does not
subscribe to `/joy` and has no manual gripper latch. It sends only validated
policy gripper commands, serializes each two-write IO pair, and reports the
last successful command to the executor. Leave `logitech_servo` stopped so
teleop cannot send competing arm or gripper commands. `joy_node` alone has no
effect on the policy gripper.


While idle, the engine continuously follows measured joints. On the first
accepted policy target it resets its filter and velocity from the current pose;
its first arm output holds that measured pose, then subsequent outputs ramp
toward the target. Both base and phased engines use this startup guard.
Do not reset or teleoperate during active policy execution.

The executor initially waits for both cameras, ordered joints, and Servo status.
`Waiting for observations: waiting for live inputs` is normal while these
subscriptions discover their publishers. Before the first action is published,
missing or stale initial inputs keep it waiting and discard any outdated queued
prediction; they do not latch a runtime fault. Once action publication starts,
stale joint feedback still latches a fault. Launch the engine before the policy
executor, and stop/restart both after an execution fault.


#### Startup jump / C153A1 protective stop

C153A1 means that the shoulder could not follow its commanded path. The earlier
base engine cached its first joint state, so resetting/repositioning the arm
between policy runs could leave an obsolete starting command. The engine now
uses fresh joint feedback at startup and blocks rather than resuming an old
trajectory after a timeout or tracking fault. This addresses a software cause;
contact, incorrect payload/center of gravity, and controller handoff can also
cause this protective stop.

After a protective stop, stop the policy executor and control engine first.
Inspect the actual cause and the robot's payload/center-of-gravity configuration
before clearing the stop using the normal pendant procedure. Prepare the
starting pose with teleop, then stop teleop and Servo as described above. Start
a fresh control engine and then a fresh policy executor. Do not bypass the stop
or increase safety limits to make the error disappear.

Startup waits for fresh joint feedback (at most 0.20 s old) and a successful
Servo-stop response. A first target more than 0.10 rad from the measured pose
is blocked. During execution, a command gap over 0.50 s, stale joint feedback,
a commanded/measured error over 0.05 rad, or an unexpected abstraction publisher
count latches `EXECUTION BLOCKED`. Restart the engine only after inspecting the
cause; later messages cannot clear the latch. These checks stop new commands;
they are not a hardware emergency stop or automatic protective-stop recovery.
Only one engine and one executor may run at a time. Both standalone engines share
an exclusive process lock; a second launch exits before creating actuator
publishers. This prevents competing gripper IO and contradictory acknowledgment
streams. If an engine has already been started in the background, use that
instance or stop it before launching another terminal instance.


While waiting for a new inference chunk, the standalone policy executor repeats
its last published seven-value target at the configured action rate for at most
2 seconds after the queue drains. It requires fresh joints and observation
inputs, and repeats the same gripper value. This prevents normal inference gaps
from falsely triggering the engine's 0.50 s command watchdog. The arm can still
converge to that last target; this is not a guaranteed physical stop. A stale
input, inference/offset fault, or exhausted wait budget stops these repeats and
latches a fault. The engine watchdog remains active if the executor stops or
crashes. Restart both processes after a latched fault; do not restart only the
engine while an old policy run continues publishing.


### Before each policy test: check automatic-control readiness

Starting the executor checks `/control_engine/status` before loading the model.
It requires an updated engine with automatic-only gripper control, no arm fault,
no other policy publisher, a stopped teleop mapper, and fresh startup feedback.
It briefly waits for initial feedback and Servo acknowledgment; an execution
fault or incompatible engine blocks startup immediately.
There are no policy-time joystick gripper buttons or manual overrides.

For a fresh pickup, reset the arm and cube to their usual starting arrangement
with no object held, then add `--prepare-episode` to the executor command. The
system commands gripper OPEN and waits for both IO writes to succeed before
making the first prediction. This is different from `/control_engine/reset`,
which sends no actuator commands and preserves the last confirmed gripper
state. Homing the arm alone does not open the gripper. Starting a new pickup
with confirmed CLOSED can lead the model toward carry/bowl behavior immediately.
Without `--prepare-episode`, startup requires an already confirmed OPEN state.

Explicit episode preparation can rearm an idle engine after an executor
publisher disconnect. It requires no policy publisher, fresh joints, Servo
stop acknowledgment, no teleop mapper and idle IO. Tracking faults and competing
source faults still require inspection and explicit reset. This preparation
opens the gripper but does not move the arm to its starting pose.

After ending a test, stop the executor and inspect any fault. To clear the
software execution latch and return the gripper to policy control without
restarting the background engine, explicitly run:

```bash
ros2 service call /control_engine/status std_srvs/srv/Trigger "{}"
ros2 service call /control_engine/reset std_srvs/srv/Trigger "{}"
```

The reset requires no policy publisher, fresh measured joints, a Servo-stop
acknowledgment, and idle gripper IO. It sends no arm or gripper command, does not
clear a robot protective stop, and does not start a policy. Wait for
`success: true` before launching the next executor. Do not launch a second
engine to clear the first engine's fault.

### 5. Policy-executor terminal — choose one execution mode

**Baseline: no Cartesian offset.**

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py --prepare-episode --speed-scale 0.5 --actions-per-prediction 10
```

**Continuous offset: translate the entire arm trajectory.**

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 10
```

**Approach-only offset: preserve pickup correction, then fade it out for carry.**

For a comparison when continuous offset grasps correctly but carry stalls, use:

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode approach --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 10
```

This mode uses the full calibrated correction for approach. After first seeing
confirmed OPEN, it requires 0.5 seconds of fresh confirmed CLOSED IO, then fades
the translation smoothly to zero over two seconds on newly planned actions.
Previously queued actions finish their checked trajectory; the transition
never clears the queue or instantly removes the correction. The log reports
`APPROACH OFFSET FADING` and `offset_scale_start`/`offset_scale_end`; a scale of
zero means the original policy joint target. MoveIt still checks these carry
targets and connecting paths. Gripper outputs and orientations are unchanged.
IO acknowledgment does not prove physical pickup. The fade is committed once
per executor launch, so restart for a new episode. This is a diagnostic option,
not evidence that continuous offset caused a carry failure. The `continuous`
mode still applies the entire correction throughout the run.

The current local file specifies **1.3 mm forward plus 0.3 mm toward wrist-image
right**, approximately +X and −Y respectively in the robot's `base` frame. Continuous mode offsets every arm target through
approach, grasp, carry, and placement. It preserves each target's orientation
and gripper value. Chunk blending happens before translation, so a target
receives the offset once. The helper loads the MoveIt robot description,
solves translated poses, and checks the interpolated joint path before
queuing the chunk. A planning/validation failure latches a fault and clears
queued actions. The extra planning work can affect timing.

**Earlier grasp-only intervention: translate immediately before first close.**

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode grasp \
  --speed-scale 0.5 --actions-per-prediction 10
```

This mode interrupts the first close request, plans the translated target,
waits for measured joint alignment, closes after fresh acknowledged open
state, and waits for close acknowledgment. It then resumes fresh policy
inference; carry/placement targets are not offset. Older trial captures used
an earlier version that paused after closing, but the current source resumes.

The offset is optional. Without `--grasp-offset-file`, both offset modes are
disabled. With a file, `continuous` is the default; specify the mode explicitly
for reproducible experiments. Do not run multiple policy executors or pair
this executor with a second controller/replay source. The standalone executor
starts its own CUDA model server; no additional server command is needed.

### Orange-fruit unseen-object tests: 20 attempts

The October 23 session uses the same frozen 110-episode, 18,000-step SmolVLA
checkpoint as the orange-cube experiment. The operator confirmed that each
official attempt contains **two orange fruits**. Each execution cycle uses
the model to pick one orange, followed by the existing checked lift-and-carry
route and taught bowl release. Full success requires both oranges in the bowl;
one orange placed is partial success. This condition measures model pickup
with assisted placement.

The executor completes one pickup/placement cycle per launch. After placing
the first orange, stop that run, return to standard pose with nothing held,
and launch the same command for the second orange. Keep the official telemetry
recording active across both cycles and report the overall outcome only after
the second cycle. This is a two-cycle assisted protocol, not an autonomous
two-object policy rollout. Keep the bowl at its taught position.

The dedicated launcher uses GPU 3 while the separate Pi0 training uses GPUs
0–2. Teleoperation and the demonstration recorder must be stopped. Keep one
control engine with bowl release running. The orange-session monitor is used
only for official evidence; it never gates practice or standard-pose movement.
The October 23 setup already started these two processes; do not start duplicates.

```bash
export UR5_VLA_ROOT="/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation"
bash "$UR5_VLA_ROOT/scripts/orange_fruit_test.sh" --check

# Physical movement: empty gripper, clear swept path. Run before each attempt.
bash "$UR5_VLA_ROOT/scripts/orange_fruit_test.sh" --prepare-start

# Same command each time; no recording announcement or marker is required to run.
# For official trials, first announce the start and wait for recording confirmation.
bash "$UR5_VLA_ROOT/scripts/orange_fruit_test.sh"
```

Use that same run command for both preliminary practice and official attempts.
It does not read, require, start or stop the monitor's recording state. The
standard-pose command is also independent of recording. Official telemetry
starts only after the operator explicitly announces an official attempt and a
manual start marker is sent. Existing evidence folders are preserved. Terminal
logs for all runs are stored separately under the session's `run_logs/` folder.
`--practice` is an alias for running the same policy without an attempt number;
an optional `--attempt N` only labels the console output. Neither option changes
official recording. The launcher sends the instruction
`Pick up the orange and place it in the bowl`, uses `--offset-mode continuous`,
`--prepare-episode`, `--speed-scale 0.5`, and `--actions-per-prediction 50`,
and enables the existing planned transport. It does not apply the temporary
blue-test pickup shift. The robot speed setting must be at most 20%.

For a manual start without chat, use this read-only recording marker before
the policy launcher (replace `1` with the intended attempt number):

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/cross_grid_trial.py" \
  --session "$UR5_VLA_ROOT/data/models/evaluation/orange_fruit_tests/session_20261023_20_attempts" \
  start 1 1
```

After the attempt, stop the policy with Ctrl+C and report the result so it can
be labelled. The monitor continues until the finish marker; alternatively,
record a complete successful attempt yourself:

```bash
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/cross_grid_trial.py" \
  --session "$UR5_VLA_ROOT/data/models/evaluation/orange_fruit_tests/session_20261023_20_attempts" \
  finish pass --pickup yes --carry yes --placement yes
```

For a failed attempt, use `finish fail` and supply the actual stage outcomes
using `--pickup`, `--carry`, and `--placement` (`yes`, `no`, or `unknown`).
This labels evidence; it never stops or moves the robot. Pickup-only outcomes
are recorded as incomplete full-task attempts with successful pickup.

Session configuration, the 20-row result table, terminal logs, telemetry,
both camera videos, snapshots and depth are stored under
`data/models/evaluation/orange_fruit_tests/session_20261023_20_attempts/`.
Internal `cross_01` means this test series, not physical table cross 1.
Times are the original measured times; the previous Excel-only 2.1 scale does
not apply. The 300-second reference flags elapsed time but does not stop the robot.
The unchanged planner still writes its plan JSON files to
`data/models/evaluation/blue_cube_tests/transport_plans/`; each executor log
records the relevant plan path.

If the engine or monitor has been stopped, source ROS and both workspaces,
then start each command in a separate terminal after checking for duplicates:

```bash
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/local_setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/local_setup.bash"

/usr/bin/python3 "$UR5_VLA_ROOT/scripts/control_engine.py" --speed-scale 0.5 \
  --bowl-release-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/bowl_release_calibration.json"

# Separate terminal; source the same ROS/workspace environment there too.
/usr/bin/python3 "$UR5_VLA_ROOT/scripts/cross_grid_monitor.py" \
  --session "$UR5_VLA_ROOT/data/models/evaluation/orange_fruit_tests/session_20261023_20_attempts"
```

### Test an unseen instruction: blue cube

The standalone executor accepts `--task`. Its default remains
`Pick up the orange cube and place it in the bowl`. To evaluate the existing
checkpoint on blue, use:

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --task "Pick up the blue cube and place it in the bowl" \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 50
```

This changes the language instruction sent with each inference request; it does
not retrain the model. Treat blue-cube performance as a new generalization test.
Use the established starting pose, an empty gripper and the calibrated bowl
position. For a simple first test, use blue as the only pickup cube. A later test
with both orange and blue present checks whether the instruction controls colour
selection. Report these conditions separately.

To open a dedicated terminal that waits for Enter before physical execution:

```bash
gnome-terminal --title="UR5 Blue Cube Test" -- \
  bash "$UR5_VLA_ROOT/scripts/blue_cube_test.sh"
```

The launcher uses the same checkpoint, pickup alignment and existing control
engine. Its current assisted mode supplies lift through the checked transport
route described below. It saves terminal output under
`data/models/evaluation/blue_cube_tests/`. Stop a running test with Ctrl+C and
inspect any fault before restarting.

For the revised blue-cube condition, add a placement-stage instruction:

```bash
/usr/bin/python3 policy_executor.py \
  --task "Pick up the blue cube and place it in the bowl" \
  --carry-task "Place the cube in the bowl" \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 50
```

`--carry-task` switches once after an OPEN-then-stable-CLOSED IO sequence
(0.5 seconds). IO confirmation does not prove a successful grasp. At the switch,
the executor discards queued pickup predictions, invalidates an in-flight pickup
response and holds the last checked target while requesting placement actions.
The checkpoint, carry lift, motion guards and bowl-release region stay the same;
the arm route remains policy-generated. Omitting this flag preserves the original
single-instruction mode.

Recorded blue attempts 1–6 used the original instruction throughout. Offline
comparisons on three post-grasp snapshots with three seeds each predicted local
motion toward the bowl for `Place the cube in the bowl` in all nine cases, while
the original pickup instruction predicted motion away in all nine. This is a
local prediction diagnostic, not nine executed trials or proof of successful
placement. Physical runs with this flag form a separate phase-conditioned
configuration. An informal physical test still overshot without placement.
The flag remains available for comparison; the current launcher uses the
assisted transport mode below.

### Assisted blue-cube transport after policy pickup

This mode keeps pickup policy-generated, then hands transport to a checked
MoveIt route after an OPEN-then-stable-CLOSED IO sequence (0.5 seconds).
Closed IO does not prove that an object was grasped. Label completion as
**assisted transport**, separately from autonomous policy generalization.

Keep the bowl at its calibrated position and use one control engine started
with the matching `--bowl-release-file`. Use:

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --task "Pick up the blue cube and place it in the bowl" \
  --planned-transport-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/bowl_release_calibration.json" \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 50
```

Omit `--carry-task`. The mode defaults policy `--carry-lift-mm` to zero because
the planned route supplies clearance. Pickup alignment remains enabled; planned
joint targets are already absolute and receive no second offset.

At handoff the executor discards queued policy predictions, invalidates any
in-flight result, and settles at its last checked target. Planning runs in a
worker while the unchanged target is held. The route lifts vertically to at
least 50 mm above the handoff pose and 80 mm above the calibrated bowl TCP
reference, moves horizontally over the reference, then targets a pose 25 mm
above the reference. Each segment must be complete. MoveIt collision/state
checks cover every resampled playback target, with joint changes limited to
0.002 rad per target and measured-feedback gating. Existing engine velocity,
acceleration, tracking and stale-feedback limits stay active.

If MoveIt's time-parameterized knots exceed the accepted spacing, the planner
bisects that Cartesian leg and replans each half, with a bounded refinement
depth. Initial discontinuities and persistent IK jumps still fail. The spacing
limit is not raised, and every final playback target is collision/state-checked.

The calibrated bowl-release gate may open the gripper once it enters the valid
region and satisfies the slow dwell. On confirmed OPEN inside the region, the
executor holds its position and never resumes the pickup policy. It faults on
stale feedback, incomplete planning, a stalled route or missing release; it does
not automatically retry or fall back to the policy. Gripper confirmation is
command evidence, not proof of a cube landing in the bowl.

MoveIt checks only registered geometry. The scene at validation contained no
world collision objects or attached cube, so checks do not establish collision
clearance against unregistered physical fixtures. Keep the scene unchanged and
supervise the first physical run. The initial validation planned complete routes
from the live pose and three recorded post-grasp poses without actuator commands.
Plans are saved under `data/models/evaluation/blue_cube_tests/transport_plans/`;
validation evidence and protocol revisions are stored with the blue session.

### Temporary pickup shift for blue attempts 12 and 13

For the October 19 session only, the operator requested an **additional 20 mm
table-forward and 20 mm table-left** shift, using the previously measured table
directions. This is a separate planned intervention before the first policy
close. The original <=7 mm continuous alignment correction stays in place.
After the model approaches, the executor discards the remaining pickup queue,
settles at the last checked OPEN target, plans the extra Cartesian translation,
plays the checked path with the gripper OPEN, waits for the endpoint to settle,
and closes. Stable CLOSED IO then triggers the existing planned bowl transport;
the bowl destination is unchanged. This does not translate the entire neural
approach and is not evidence of autonomous pickup generalization.

First start the corresponding trial recording, then run:

```bash
bash "$UR5_VLA_ROOT/scripts/blue_cube_test.sh" --attempt 12
# For the next recorded trial instead:
bash "$UR5_VLA_ROOT/scripts/blue_cube_test.sh" --attempt 13
```

The terminal waits for Enter. The launcher's extra executor arguments are:

```bash
--pickup-shift-file "$UR5_VLA_ROOT/data/models/evaluation/blue_cube_tests/session_20261019_20_attempts/temporary_pickup_shift_12_13.json" \
--experiment-attempt 12
```

Use `13` for the second trial. These flags require the matching active recorder
trial and `--planned-transport-file`; attempts outside 12/13 are rejected.
Omitting the flags restores the ordinary policy command, including for attempt
14 onward. The special mode accepts at most 30 mm total translation and 1 mm
vertical translation. It does not relax the existing small-offset or tracking
limits. The measured table basis produces an additional base-frame vector of
approximately `[0.0200931, 0.0199700, 0.0000252]` metres.

For an explicitly requested **unrecorded demonstration** of the completed
attempt-13 condition, keep the recorder waiting and use:

```bash
bash "$UR5_VLA_ROOT/scripts/blue_cube_test.sh" --repeat-13-unrecorded
```

This selects the same pickup shift and transport settings, requires formal
attempt 13 to be finished, and does not create a trial marker or modify its
result/configuration. Camera and telemetry trial recording stay inactive, and
the launcher does not save a terminal transcript. Checked planning diagnostics
may still be saved by the motion helpers. The terminal waits for Enter; reset
the physical scene and start pose first. All motion/feedback checks remain in
place. Return to the ordinary command for the next numbered experiment.

Every exact playback target is checked with MoveIt, at <=0.002 rad joint steps.
Planning, stale feedback, unexpected closure, endpoint settling and close
acknowledgment failures stop the intervention without resuming the pickup
policy. Read-only validation produced complete paths from the current pose
and recorded pre-close poses in attempts 7, 10 and 11. The registered scene
contained no world objects or attached cube: this does not establish physical
fixture clearance. Supervise these two physical tests. Runtime configuration
and code hashes are saved in each trial's `configuration` directory, and plans
under `data/models/evaluation/blue_cube_tests/transport_plans/`.

### Recorded evaluation report and Excel results

`scripts/build_blue_cube_report.py` builds a PDF, editable Word report, Excel
workbook and a document ZIP from the completed October 19 session. Use a Python
environment with `openpyxl`, `python-docx`, `reportlab`, `matplotlib` and `Pillow`:

```bash
python scripts/build_blue_cube_report.py \
  --session data/models/evaluation/blue_cube_tests/session_20261019_20_attempts \
  --scene-confirmation data/models/evaluation/blue_cube_tests/session_20261019_20_attempts/scene_confirmation.json \
  --output /home/grasplab/Downloads/UR5_20_Attempts_Report
```

The selected series has 8 reported physical completions, 5 partial completions
and 7 other failures. The operator confirmed **19 blue-cube trials (7 complete;
36.8%) and one orange-cube comparison (complete)**. Attempt 20's saved command
still requested blue, so its orange-cube completion is not evidence of correct
colour selection. Overall 8/20 is a mixed-scene, mixed-execution system result.
Complete passes used planned transport; the workbook identifies execution
conditions and the temporary pickup intervention. Two software-aborted
attempt-19 captures, premature starts and unrecorded demonstrations are excluded
and explicitly audited. Capture durations are not verified task durations.

The workbook contains per-attempt outcomes, every stored summary value, joint
tracking metrics, rates, gripper/IO events, provenance links and an index of 400
real candidate frames. Full per-message data remain in the original JSONL and
video files. Downloaded documents do not include those large raw streams.

Package all 400 unchanged exported PNGs and a separate 100-image partial-success
subset (attempts 3, 4, 5, 6 and 9), with camera contact-sheet PDFs:

```bash
python scripts/package_trial_camera_images.py \
  --session data/models/evaluation/blue_cube_tests/session_20261019_20_attempts \
  --output /home/grasplab/Downloads/UR5_Camera_Images
```

The ZIPs preserve original exported image bytes and include timestamps, source
manifests and SHA-256 indexes. PDF images are smaller previews. Each attempt
has ten base and ten wrist views; review individual frames before assigning
pickup or placement captions. This packager requires `Pillow` and `reportlab`.

### Blue-cube RGB-D localization and calibration

The frozen pickup policy can approach a familiar location instead of following
the blue cube. A separate **read-only** detector is available to investigate
this. It does not change the policy, publish robot targets, or move the gripper.
Run from the repository root with the blue cube alone on a clear table cross:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/blue_cube_localization.py \
  --output data/models/evaluation/blue_cube_tests/vision_localization_check \
  --duration 600 --show
```

Press `q` in a preview window to close it. Annotated `base_latest.jpg` and
`wrist_latest.jpg`, `latest.json`, and append-only `observations.jsonl` are
saved in the output folder. An optional `--base-roi X0 Y0 X1 Y1` restricts the
base view to a measured image region. The detector rejects stale or unsynchronized
RGB/depth, clipped blue regions, missing depth, and multiple candidates. Color
segmentation identifies blue regions; a unique candidate is not proof of a cube.
The reported XYZ is the **visible surface in camera optical coordinates**, not
the robot target or finger grasp centre. Validate detection across cube positions
before treating it as a localization method.

The October 19 probe found both cameras using `camera_color_optical_frame`
despite separate topic namespaces, with no transform connecting that frame to
robot `base`. Do not use this ambiguous TF tree for pickup. Configure unique
camera frame identities during a planned camera restart, then calibrate the
fixed base camera to robot `base`. Wrist localization additionally needs a
validated hand-eye transform evaluated at the image timestamp. A calibrated
grasp centre relative to `tool0` and table/cube geometry are also required.

Generate a printable checkerboard with this command (requires `reportlab`):

```bash
/usr/bin/python3 scripts/create_calibration_target.py \
  --output /home/grasplab/Downloads/UR5_Calibration_Targets
```

Choose the A4 or Letter PDF matching your paper and print just that file at
**100% / Actual size**, with Fit and Shrink disabled. The pattern has 9 by 7
squares, each 20 mm, giving **8 by 6 inner corners** for OpenCV. Measure the
100 mm reference bar and a square after printing, then mount the sheet flat on
rigid matte backing with its white border intact. Keep its TOP LEFT orientation
label identifiable. PDFs were rasterized and all 48 inner corners detected in
both paper variants; printed dimensions still require a ruler check. Printing
the target does not establish camera-to-robot calibration; define the physical
capture arrangement before attaching it to the arm or collecting robot poses.

Check the mounted board in both cameras and record initial references with:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/checkerboard_capture.py \
  --output data/models/evaluation/camera_calibration/checkerboard_session \
  --duration 600 --show
```

The read-only windows display detected corners. Press `s` to save a paired
reference when both cameras detect all 48 corners and the robot pose at the
wrist image timestamp is available; press `q` to exit. Keep the arm stationary
for a reference. The base camera may need the rigid board tilted toward it to
resolve all corners. Samples preserve original PNGs, corner measurements,
camera intrinsics, source timestamps, reprojection error, and `base` to
`tool0_controller` pose. Image processing runs separately from ROS callbacks.
These snapshots are not automatically accepted calibration measurements:
checkerboard orientation and stationarity remain explicitly unverified.
Camera mounting, board placement and the multi-pose capture arrangement must
be established before solving hand-eye or camera-to-robot calibration.

For wrist-only collection, keep the checkerboard fixed and the wrist camera
rigidly mounted. After stopping the policy executor, control engine and normal
joystick mapper, use the already active forward position controller and start
Servo. Do not run a standard-pose reset as part of this capture:

```bash
source /opt/ros/humble/setup.bash
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
/usr/bin/python3 scripts/calibration_teleop.py
```

This special mapper enables physical motion only after seeing the trigger
released, then held. Trigger plus stick gives base-frame translation at up to
15 mm/s per axis; buttons 5/6 move down/up. Hold **physical button 4 plus the
trigger** to use the stick for X/Y tilts instead, at up to 0.10 rad/s per axis.
Handle twist controls Z rotation. The gripper and standard-pose buttons are
disabled. Trigger release or joystick data older than 250 ms commands zero
Twist. A competing mapper, policy source or control engine latches this mapper
off. These are software controls; retain access to the pendant stop.

In a second terminal, start the read-only recorder:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/checkerboard_capture.py --camera wrist \
  --output data/models/evaluation/camera_calibration/wrist_only_session \
  --duration 3600 --show --auto-save
```

Move manually between clear views, release the trigger, and wait at least two
seconds with all corners visible. Up to 20 distinct views are saved when the
image-time tool pose is available, reprojection RMS is below 1 pixel, and tool
drift over 0.5 s is below 0.5 mm and 0.15 degrees. A pose must differ from every
saved pose by at least 15 mm translation or 8 degrees rotation. These thresholds
are collection gates, not calibration accuracy guarantees. Include tilts about
two nonparallel axes; translation and yaw alone are insufficient. The samples
still need consistent physical checkerboard orientation review, excitation
checks, hand-eye solving and independent validation before any vision-guided
pickup. Wrist-only mode does not create a base-camera transform.

`scripts/fit_camera_extrinsics.py` fits a rigid transform from **measured** 3D
correspondences and checks a separate set of held-out positions:

```bash
/usr/bin/python3 scripts/fit_camera_extrinsics.py /path/to/measured_points.json \
  --output /path/to/new_camera_calibration.json
```

The input needs `camera_serial`, `camera_namespace`, `source_frame`,
`target_frame` equal to `base`, and four arrays of XYZ points in metres:
`camera_points_m`, `robot_points_m`, `validation_camera_points_m`, and
`validation_robot_points_m`. Each camera/robot pair must describe the **same
physical point**. Provide at least four fit points and four distinct validation
points spread across the workspace. Pairing an uncalibrated flange position
with a cube surface is invalid. Maximum fit and held-out error must both be
within 5 mm; tighter grasp tolerances require further validation. The output
always leaves robot motion disabled. Automatic vision-guided pickup is pending
physical calibration and checked motion integration, and must be recorded as a
separate experiment condition from frozen-policy trials.

### Slower execution profile

The standalone executor and base control engine now default to **0.5 speed
scale**. Use the same `--speed-scale` on both processes. The engine retains
125 Hz command publication but halves velocity/acceleration limits and doubles
its filter time constant to 0.56 seconds. The executor plays actions at 10 Hz
instead of the recorded 20 Hz. It executes only the first 10 of each 50-action
prediction and discards the other 40, with a small prefetch window, so a full
chunk is not stretched over five seconds without another observation.

| Parameter | Default | Meaning |
| --- | --- | --- |
| Engine `--speed-scale` | `0.5` | Multiply 0.10 rad/s velocity and 0.30 rad/s² acceleration limits; accepted range is greater than 0 through 1 |
| Executor `--speed-scale` | `0.5` | Multiply nominal 20 Hz action playback rate |
| Executor `--actions-per-prediction` | `10` | Execute the first N predicted actions, then reobserve; accepted range 1–50 |

For a slower 0.25 profile, set `--speed-scale 0.25` on both commands; the
engine limits become 0.025 rad/s and 0.075 rad/s² and playback becomes 5 Hz.
The shorter action horizon does not alter model weights or the recorded
training rate. It changes deployment timing and feedback frequency. Reduced
speed may help tracking and overshoot; it does not establish improved model
accuracy, correct a camera/tool calibration error, or guarantee a grasp.
At very low scales the grasp-only mode may hit its alignment/closure timeouts.

For an explicit return to the earlier full-chunk playback settings, use
engine `--speed-scale 1.0` and executor `--speed-scale 1.0
--actions-per-prediction 50`. Safely stop/reset and restart both processes when
changing profiles. These standalone profile arguments do not change the
package's shadow/offline commands or the phased policy executor's own timing.

### Offset configuration and changing its size

Display the current local configuration:

```bash
cat "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json"
```

The helper reads `offset_frame` and `offset_m`; the latter is a three-element
translation vector in **metres**. `requested_offset_mm` is a descriptive
record, not the value consumed by the planner. For example, exactly +X at
0.6 mm would be `"offset_m": [0.0006, 0, 0]`. The lab's measured vector has
small Y/Z components retained from the forward calibration jog. The current
configuration uses a 1.3 mm forward component and a separately measured
0.3 mm wrist-image-right component:
`offset_m` is approximately `[0.0012985963, -0.0003019297, 0.0000036881]`.
`offset_components_m` preserves the forward and right contributions for later
edits; the planner consumes their already-summed `offset_m`. Right was measured
by a 15.47 mm manual translation, mainly −base Y, with a 0.29° orientation
change. This is a fixed base-frame correction measured at the calibration pose,
not a camera-frame direction that rotates with the wrist. The calibration
evidence is in `data/models/evaluation/codex test analysis/right_offset_calibration.json`.
Offset magnitude
must not exceed approximately 7 mm. Do not use a joint-angle offset as a
substitute for this Cartesian correction.

The configuration is loaded at startup. Stop, safely reset/release, change
the file, and restart the executor to change the offset. Do not restart while
startup gripper commands could release a held object. `data/` is ignored by
Git: a fresh clone does not include this calibration file, trial recordings,
or model weights. Supply the complete checkpoint and a measured configuration
for that physical setup; the lab's meaning of “forward” is not universal.

### Logs and read-only runtime diagnostics

The executor fetches the live MoveIt robot description during preflight,
before model loading and camera subscriptions. A lost parameter reply gets one
bounded retry, then startup refuses to proceed. The in-memory description is
verified against MoveIt FK before offset use; path checks remain mandatory.
No persistent URDF cache or unchecked fallback is used.

For a controlled comparison with the original full-chunk executor, retain the
current continuous offset and 0.5 speed and use all 50 predicted actions:

```bash
/usr/bin/python3 policy_executor.py \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 50
```

This comparison keeps the later 40 predictions that the shorter horizon drops.
At 0.5 speed a 50-action chunk covers about 5 seconds, so observations are used
less frequently for new predictions than with 10 actions. Prefetch is 12 for
50 actions and 2 for 10; offset geometry and controller limits are unchanged.
This is a diagnostic comparison, not proof that horizon length caused a stall.
Prepare the starting scene with no object held; `--prepare-episode` opens the
gripper before inference. A normal end-of-executor fault can be rearmed by this
explicit new-episode preparation, while other faults still require inspection.

If pickup succeeds but carry stalls, check both the control engine and executor
terminals before attributing it to the offset. `CONTINUOUS OFFSET CHECKED`
means the checked path was accepted; an offset path rejection reports a specific
fault. `POLICY DIAGNOSTICS` compares joint spans in the full 50-action prediction
and the selected execution horizon, the first predicted target against the
observed joints, and observed/predicted gripper commands (0 closed, 1 open).
Small predicted spans after pickup suggest a policy that keeps predicting a
nearby pose; substantial targets with an engine fault instead point to a
control/feedback problem. The engine prints `CONTROL DIAGNOSTICS` every two
seconds: target/measured gap, sent-command/measured gap, commanded speed, and
automatic gripper-control mode. A large target gap with low command tracking error
means the robot follows the smoothed command but has not reached the policy
target; compare commanded speed with the configured velocity limit. These
diagnostics do not verify physical grasp or
object contact. Do not disable path or tracking checks to force a carry motion.


Useful messages and their meaning:

| Message | Interpretation |
| --- | --- |
| `Queued 50 policy actions...` | Predictions were queued; not evidence of a successful task |
| `Waiting for observations...` | Missing, stale, or invalid camera/joint/gripper/Servo input |
| `CONTINUOUS OFFSET ENABLED` | Whole-trajectory correction selected at startup |
| `CONTINUOUS OFFSET CHECKED` | A translated chunk passed planning/path checks |
| `CONTINUOUS OFFSET BLOCKED; fault latched` | Continuous correction failed; restart only after diagnosing/resetting |
| `ONE-GRASP ASSIST ENABLED` | Grasp-only intervention selected |
| `OFFSET PLAN CHECKED` / `OFFSET REACHED` | Grasp correction planned / measured target alignment reached |
| `ASSISTED GRASP FINISHED` | Close command acknowledged; fresh policy inference resumes, not proof of a held cube |
| `Rejected arm target jump...` | Engine rejected a target relative to its retained reference |
| `Arm output blocked: expected 1 abstraction publisher...` | Engine detected zero or multiple abstract-command publishers |

MoveIt service discovery is asynchronous. Offset planning now waits briefly
for each required service instead of rejecting a newly created client
immediately. If a service remains unavailable, the error includes its full
name. Keep MoveIt running in the same ROS domain and restart the executor
after resolving a latched fault.

In a separate prepared terminal, run these checks one at a time. Stop each
`topic hz` command with Ctrl+C after inspecting its rate:

```bash
ros2 topic hz /control_abstraction_layer
ros2 topic hz /forward_position_controller/commands
ros2 topic hz /joint_states
ros2 topic echo /control_engine/gripper_state --once
ros2 topic info /control_abstraction_layer --verbose
ros2 topic info /forward_position_controller/commands --verbose
```

A registered publisher is not necessarily publishing: a stopped Servo node
may remain listed. Check actual rates and source state, and keep the mapper
stopped. The default target rates are 10 Hz policy output and 125 Hz engine output.
The model still predicts 50 actions, but only the first 10 are played over
approximately one second before requesting another observation. Inference
latency/prefetching can increase observation age. Rapid gripper close/open predictions can still prevent a
stable grasp. The model does not independently detect pickup/placement success
or stop automatically when the task appears complete.

The control engine now requires a changed gripper request to persist for
0.2 seconds before starting its serialized IO transaction. Requests separated
by more than 0.25 seconds do not accumulate confirmation time. This applies to
both opening and closing, rejects brief policy reversals, and permits sustained
retry/release commands. It does not lock the gripper closed or infer contact.
The explicit `--prepare-episode` opening remains immediate after readiness checks.
`/control_engine/status` reports `gripper_stability_s` and `gripper_pending`.
Restart the engine to load this change; existing engine processes keep their
previous behaviour. Keep one engine only and stop the executor before restarting
the engine. This changes the evaluation configuration: cross 1 and cross 2
attempts 1–3 used the previous unfiltered gripper logic. Record later attempts
as a different configuration if the new engine is loaded.

With `--bowl-release-file`, the first confirmed closed IO at least 25 mm from
the release point now latches a transport hold. Policy open predictions cannot
reopen the gripper until the measured bowl-release condition triggers. This
prevents the observed repeated opening after a successful first grasp, but also
disables automatic grasp retries: IO cannot distinguish a held cube from an
empty closed gripper. A missed grasp requires stopping and preparing a new
episode. Missing TF preserves this hold and cannot force release.
`BOWL DIAGNOSTICS` logs and `/control_engine/status` expose release distance,
XYZ displacement, orientation error, TF age, speed, and dwell progress. If
release fails, inspect these measurements before changing the release region.

The stability check cannot shorten a delay during which the model continues
requesting closed. It does not force a timed release: a reliable automatic
bowl-release override requires a verified release region and confirmation that
the tool has reached it. Gripper IO acknowledgments alone do not establish
object contact or bowl arrival.

The executor refreshes the gripper observation buffer on every actual
`/control_engine/gripper_state` confirmation, including while inference and
MoveIt checks are running. Inference scheduling does not refresh a cached
gripper value's timestamp. This fixes the false stale-gripper fault seen in
cross 6 attempt 3 while retaining rejection of genuinely missing feedback.

Joint feedback uses a separate callback group and a two-thread ROS executor;
action and service callbacks in the default group remain serialized. Carry
observation FK runs in the inference worker, and IK yields between targets.
Safety checks read a coherent joint-position/timestamp pair. These changes
address a local stale-joint fault seen during cross 4 attempt 1 without changing
the 100 ms executor freshness threshold or engine watchdog. Stale-joint errors
include the observed age. A read-only live callback stress check passed, but
intermittent system or driver delays can still stop execution.

After simultaneous camera/Servo staleness recurred, all base/wrist images,
joint states, gripper acknowledgments and Servo status subscriptions were moved
to `bypass_policy_observations`, a read-only node spun by its own executor
thread. Policy actions have a separate mutually exclusive timer group so
MoveIt service responses cannot serialize sensor delivery or action ticks.
The prior joint-only separation was insufficient for the other sensors.
An eight-second read-only stress test blocked the policy's default callback
group for 800 ms every 900 ms: camera ages stayed below 98 ms, joint ages below
14 ms, and intercepted action gaps below 107 ms. No command DDS publisher was
created by that test. Freshness limits remain unchanged; physical validation
is still required after restarting the executor.

For the fixed bowl calibrated on October 15, enable the measured-pose release
check in the **control engine terminal** (keep the executor stopped during startup):

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 control_engine.py \
  --bowl-release-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/bowl_release_calibration.json"
```

Then use the normal executor command with `--offset-mode continuous`,
`--prepare-episode`, `--speed-scale 0.5`, and `--actions-per-prediction 50`.
The optional check first requires confirmed closed IO for at least 0.5 seconds
at least 25 mm away from the release reference. The original calibration used
a 10 mm / 15 degree pose check. Live camera inspection showed the policy correctly
holding the cube over the bowl about 84 mm from that manually taught pose, so
the current calibration instead uses the camera-confirmed policy arrival as
its reference. Following the successful cross 2 attempt 4, the operator requested
30 mm horizontal displacement, then a more lenient vertical range of 30 mm
below through 120 mm above that reference; the current `release_zone` uses
a 40 mm horizontal radius (requested after cross 6 attempt 2), retains the
above vertical limits, and allows
30 degrees of orientation variation. A trial with 50 mm radius released near
the rim and dropped the cube outside (cross 3 attempt 5), so it was reverted
to 30 mm before the later 40 mm adjustment.
These limits are configured in the calibration JSON and do not
require a precise single release height. The engine opens once when fresh
`base -> tool0_controller` feedback stays inside this region, moving at no
more than 20 mm/s, for 0.3 seconds. Missing/stale
TF cannot trigger release. The gripper remains requested open after this
one-shot release until explicit episode preparation/reset, preventing repeated
closing over the bowl. Arm targets and the 1.3 mm forward / 0.3 mm right offset
are unchanged. Closed IO is command confirmation, not proof of a held cube.
This feature has offline tests; physical release performance still needs testing.
Recalibrate if the bowl or desired release pose moves. Omit the flag to disable
the pose-based override. Do not mix these assisted results with the earlier
unassisted baseline when calculating success rates.

At the operator's request after cross 6 attempt 2, corrected carry lift is
**25 mm by default for continuous mode with a grasp-offset file**. Its first physical trial
(cross 6 attempt 1) stopped at the target-step guard after lift activation.
The wall-time ramp was replaced with a ramp from the committed queue endpoint;
joint observations are compensated for the applied lift before inference to
avoid repeatedly adding height. Use `--carry-lift-mm 0` to return to the
previously working motion without lift. The current command, with height
specified explicitly for reproducibility, is:

```bash
/usr/bin/python3 policy_executor.py \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous --prepare-episode \
  --speed-scale 0.5 --actions-per-prediction 50 \
  --carry-lift-mm 25
```

This lift remains zero during approach. `--carry-lift-mm 0` also disables it.
Other modes default to no lift. After open IO has been seen
and closed IO stays confirmed for 0.5 seconds, newly planned targets ramp upward
in base Z over two seconds of planned action time, reaching 25 mm. Previously queued targets finish
first. Height remains raised for the rest of that executor episode, including
release; restart the executor for a fresh pickup. The bowl's vertical release
range already permits the extra height. This does not detect successful contact
or guarantee that learned targets reach the bowl.
The option accepts 0–30 mm and requires continuous offset mode. The ordinary
grasp correction retains its 7 mm translation bound; the explicitly requested
carry component is separate. Its relative joint-correction envelope expands
with the requested height, while measured-start/consecutive-target checks,
joint limits, and interpolated MoveIt collision checks still apply. Rejected
paths stop execution. A read-only live-model 25 mm preview passed 50 MoveIt
checks without sending actuator commands; physical carry-height performance
still needs a supervised trial. Record this as a new evaluation configuration.

To retain terminal output for a supervised run, replace the executor command
with a logged version in its terminal:

```bash
mkdir -p "$UR5_VLA_ROOT/data/models/evaluation/policy_runs"
UR5_RUN_LOG="$UR5_VLA_ROOT/data/models/evaluation/policy_runs/executor_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
/usr/bin/python3 -u "$UR5_VLA_ROOT/scripts/policy_executor.py" \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous \
  --speed-scale 0.5 --actions-per-prediction 10 2>&1 | tee "$UR5_RUN_LOG"
```

This writes a terminal log, not a synchronized camera/telemetry recording.
The earlier monitored tests and their separate analyses are stored locally
under `data/models/evaluation/codex test analysis/test_01/` through `test_06/`.
Changed model/controller code and offset values must be noted when comparing
runs; several captures include restarts or multiple attempts.

### Shutdown, reset, and current limits

The standalone executor runs checked command publication on its own ROS node
and single-thread executor, separate from inference/MoveIt service callbacks
and sensor callbacks. Attempt 19 exposed a 718 ms command gap around pickup
handoff: the engine's 500 ms watchdog stopped output first, then the carry
helper reported that measured joints were not following. A carry-tracking
message can therefore be secondary to an earlier engine timeout; inspect the
engine log and command timestamps before changing motion limits. The isolated
executor was checked with two main workers deliberately blocked for 750 ms:
checked wait publication continued at approximately 100 ms intervals. This is
a software scheduling test, not a hard real-time guarantee or physical retest.
Feedback freshness, fault latches and the engine watchdog remain enforced.

Orange-fruit practice exposed a separate planning-service timeout. A no-motion
test with the production timers reproduced it; short standalone service tests
had missed it. MoveIt clients now use `planning_service_node.py`: a separate
ROS context and a serialized, caller-driven single-thread executor. Request
and response handling stay together, without a background response thread or
an extra completion-notification callback. Each state still requires an
affirmative MoveIt result; unanswered requests and invalid states fail closed.
The revised production-timer test ran for 300 seconds with 3,038 planning
requests, zero response timeouts and all command output intercepted. The full
561-target bowl route was also checked without execution. These checks are
software validation, not a physical task-success result. Evidence is in the
orange session's `setup/` directory.

The first physical attempt-19 retest still had a 736 ms gap, despite the
dedicated command executor. A further change removes synchronous ROS/terminal
diagnostic writes from command callbacks and the transport helper: diagnostics
are queued and flushed on the main node. The handoff releases the action-queue
lock before calling the transport helper. A regression test exercises the actual
handoff wrapper while its diagnostic backend sleeps for 750 ms; checked hold
publication continues. This removes a blocking path, but does not prove that
logging caused the captured gap or establish hard real-time behavior. The
additional change needs a supervised physical retest. Do not increase the
watchdog to conceal these interruptions.

Stop the policy executor with Ctrl+C, then stop the control engine with Ctrl+C.
The executor attempts to terminate its model-server subprocess on exit. Secure
any held object before operating the gripper or changing controllers. Software
process shutdown is not a certified immediate robot stop; use the pendant/E-stop
if motion must stop urgently. To return to teleop, leave both standalone
processes stopped, run the guarded start-pose preparation when appropriate,
and restart the mapper. To return to trajectory control:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller
ros2 control list_controllers
```

At the default 0.5 scale, the base standalone engine limits velocity to
0.05 rad/s and acceleration to 0.15 rad/s². At scale 1.0 its limits are
0.10 rad/s and 0.30 rad/s². Its startup gate permits at most 0.10 rad
from fresh measured joints; consecutive target jumps above 0.15 rad are blocked.
A 0.20 s feedback timeout, 0.50 s command timeout, and 0.05 rad tracking-error
gate latch a fault and suppress further arm and new gripper commands. Rejected
arm commands no longer trigger gripper transitions. Dry-run performs no Servo
stop calls or actuator writes. The engine still does not require a physical
deadman, monitor all robot safety/controller states, or arbitrate publishers
writing directly to the forward controller. Keep teleop, Servo and other direct
command sources stopped throughout policy execution.
The current MoveIt scene omits the table, bowl, cameras, and people; passing
its collision checks does not certify the real swept path. This experimental
workflow requires supervision and is not equivalent to the separately guarded
package live/episode executors.

## Typical read-only checks

```bash
ros2 pkg prefix ur5_vla_dataset_recorder
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" --help
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" offline
cat "$UR5_VLA_ROOT/data/dataset/lerobot/ur5_cube_110ep_v3/meta/info.json"
```

Recording is launched with `ros2 launch ur5_vla_dataset_recorder recorder.launch.py` after the robot and cameras are ready. Its output goes to the repository data folder. The live control and policy scripts above can move the robot; they were **imported and syntax-checked**, not executed on hardware during this migration.

## Historical material and known limits

`dump/` preserves earlier simulation, bottle, lemon, and cube experiments. The archived `dump/vla_script_training/convert_data*.py` and `conversion_data_cube_twoCam.py` scripts now look under `dump/raw/dataset_raw_cube` or `dump/raw/dataset_raw_bottle` and write under `data/hf_lerobot_cache`; they refuse to overwrite an existing converted dataset. Those earlier CSV raw directories were not present on this workstation, so the scripts cannot complete until the matching inputs are supplied. The current `frames.jsonl` raw dataset has a different schema. Archived training logs and notebook outputs may include old machine paths; they are historical observations, not current configuration. The dated package handoff document has the same status.

ROS hardware drivers, Conda, CUDA, Google Drive, and the physical arm remain external dependencies. A one-frame GPU inference completed from the copied checkpoint and dataset. That validates local model loading and an offline prediction; it does **not** establish physical task success or certify live motion safety. No live control process was started by this migration.

## Verification and Git

### ROS executable smoke test (2026-10-11)

All 16 installed console executables from `ur5_vla_dataset_recorder` and
`ur5_isaac_simulation` were discovered from the new workspace overlays.
The `serial` and Robotiq packages provide libraries, plugins, descriptions,
and launch files; they do not install additional `ros2 run` executables here.
ROS nodes were started only in isolated domain 210 with
`ROS_LOCALHOST_ONLY=1`; no physical robot command was sent. A timeout exit
code of 124 means the node remained running until deliberately stopped.

| Executable | Result | Limit |
| --- | --- | --- |
| `dataset_recorder` | Started; `/dataset_recorder` and start/stop services visible | Recording with live cameras and arm untested |
| `logitech_servo` | Started; warned that SetIO was absent in isolation | Joystick, gripper IO, and arm response untested |
| `move_to_standard_pose`, `prepare_dataset_start` | Started and waited for robot state/services; interrupted after five seconds | Physical movement and safety checks untested |
| `policy_executor`, `policy_live_executor`, `policy_episode_executor` | `--help` passed | Live action execution untested |
| `convert_lerobot`, `validate_lerobot`, `package_lerobot_dataset`, `install_lerobot_checkpoint`, `split_lerobot_archive` | `--help` passed | Dataset operations were not repeated in this smoke test |
| `offline_policy_check` | Fails under system Python because `torch` is missing; module `--help` passes with `pi0` Python | Run as `/home/grasplab/miniconda3/envs/pi0/bin/python -m ur5_vla_dataset_recorder.offline_inference ...` after sourcing overlays |
| Isaac `ur5_controller`, `gripper_controller` | Started and remained active until timeout | Both print an `rcl_shutdown already called` traceback on interrupt; action execution untested |
| Isaac `ur5_isaac_ros2` | Fails at import: `tf_transformations` missing | Install `ros-humble-tf-transformations` in the host ROS environment, then retest with Isaac Sim/display available |

The Isaac package currently omits its `tf_transformations` runtime dependency
from `package.xml`; the host ROS package is not installed. The Isaac node
also uses a GUI, so a headless smoke test cannot establish complete startup
even after the dependency is installed. A supervised live test is still needed
for physical nodes when the robot, controller state, speed slider, cameras,
and swept path are ready.

On 2026-10-10: both workspaces built; the recorder package resolved from this repository in a fresh shell; `pi0` imported LeRobot and the UR5 package from here; 67 system-Python package tests passed with one skipped, and six model-protocol tests passed in `pi0`; Python compilation, shell syntax, and notebook JSON checks passed; one-frame GPU inference completed with only the new cache locations for the canonical 110-episode checkpoint and for both downloaded checkpoints selected by the copied policy scripts; both copies of the thesis chapter builder completed and refreshed 57 evidence entries. The 95 standalone files copied from the earlier working folder matched their originals byte-for-byte at the time of copying. Source data, checkpoint, and cache copies were checked with rsync dry runs.

`.gitignore` excludes data, model weights, older generated outputs, upstream LeRobot checkouts, Python caches, and both ROS build trees. Source code, notebooks, reports, and this README remain available to commit. No commit or push was made as part of this migration.


## Numbered-cross evaluation: five attempts per location

The 2026-10-15 evaluation uses 20 cross centres and five attempts per centre
(100 attempts). Locations are numbered from top to bottom in the reference
photo, left to right within each row. The original photo, map coordinates and
numbered reference are stored under
`data/models/evaluation/codex test analysis/cross_grid_20261015/reference/`.
These are photo coordinates; they do not establish calibrated base-frame
coordinates for a cube. Keep physical markings, bowl position and cube
orientation consistent across attempts, and freeze the execution profile.

The chosen limit is **300 seconds**, measured from the operator's start marker;
first policy/arm command times are also recorded to distinguish startup delay.
A reported pass means pickup, carry and release into the bowl without human
assistance. Pickup-only, carry and placement can be recorded separately. The
monitor flags the time limit but sends no stop or motion commands. A reported
pass after 300 seconds is preserved as an operator pass and separately marked
as outside the limit.

The monitor subscribes to measured joints/velocities, policy targets, forward
controller commands, gripper IO/acknowledgments, robot/safety/Servo status,
speed scaling, and runtime logs. It records actual TCP poses at 10 Hz, two
colour videos at nominal 10 FPS with source timestamps, 1 Hz review snapshots,
lossless depth snapshots when available, camera intrinsics, MoveIt model
identity, topic publishers, controller inventory and per-attempt configuration
snapshots. IO confirmation does not prove object contact. Pose comparisons are
between commanded and measured robot states, not errors relative to the cube.

Only one read-only monitor should run for this session:

```bash
source /opt/ros/humble/setup.bash
cd "$UR5_VLA_ROOT"
/usr/bin/python3 scripts/cross_grid_monitor.py \
  --session "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/cross_grid_20261015"
```

The assistant can place start/finish markers when the operator says
`cross 01, attempt 1, start` and then reports the result. Manual marker commands
are also available; they affect recording only:

```bash
cd "$UR5_VLA_ROOT"
/usr/bin/python3 scripts/cross_grid_trial.py \
  --session "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/cross_grid_20261015" \
  start 1 1

/usr/bin/python3 scripts/cross_grid_trial.py \
  --session "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/cross_grid_20261015" \
  finish pass --pickup yes --carry yes --placement yes
```

Each attempt is kept in `cross_XX/attempt_YY/`, with `observations.jsonl`,
`summary.json`, `trial.json`, `analysis.txt`, `video/`, `camera_snapshots/`,
`depth/` and `configuration/`. Existing attempt folders are never overwritten.
`results.csv` and `aggregate.json` track per-location outcomes. An experiment
change should be recorded as a protocol deviation rather than silently mixed
with the same baseline. Evaluation files are local ignored data; preserve them
separately from a source-code Git push.
