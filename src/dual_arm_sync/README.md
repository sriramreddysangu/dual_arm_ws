
# dual_arm_ws

Author: [@sriramreddysangu](https://github.com/sriramreddysangu), IIT Tirupati

ROS 2 workspace for **synchronized, collision-free trajectory generation for multiple Doosan M1013 arms** sharing a workspace, planned in joint space and executed in Gazebo.

The workspace contains one ament_python package, [`dual_arm_sync`](src/dual_arm_sync/), which implements a staged pipeline:

```
IK + goal selection -> B-spline seed -> collision scan -> Kuramoto phase synchronization
                                                                  |
                                                  (if timing alone is not enough)
                                                                  v
                          B-spline deformation (perpendicular escape) + re-sync -> Gazebo execution
```

It scales from 2 arms up to a 9-arm cell, uses real collision meshes of the M1013 (via FCL) with a capsule fallback, and includes closed-loop benchmark scripts that plan and execute randomised trials.

Reference implementation of *"Kuramoto Synchronized Configuration Space Trajectory Generation for an Ensemble of Manipulators Sharing a Common Workspace"* (ICRA 2027 submission, under review). Interference between arms is resolved **temporally first** (a repulsive Kuramoto phase-oscillator model along unmodified B-spline paths) and **geometrically only where timing is insufficient** (perpendicular escape injected as B-spline control-point displacements, then a full re-synchronization). A paper-to-code map, the settings needed to reproduce the paper's experiments, and the known differences between the paper and the code are in [`src/dual_arm_sync/README.md`](src/dual_arm_sync/README.md).

## Repository layout

```
dual_arm_ws/
├── README.md                  <- this file
└── src/
    └── dual_arm_sync/         <- the ROS 2 package (see its README for details)
        ├── dual_arm_sync/     <- Python modules and step_* pipeline scripts
        ├── launch/            <- Gazebo / RViz / MoveIt launch files
        ├── config/            <- pipeline_params.yaml, sample_workspace_config.yaml
        ├── worlds/            <- Gazebo world
        ├── rviz/              <- RViz config
        └── test/
```

`build/`, `install/` and `log/` are git-ignored.

## Reproducing the paper

The Table III experiments (2, 4 and 6 arms on a 1.39 m grid, 150 trials each) run with `step_107`, `step_117` and `step_127`. Several code defaults differ from the paper's Table II; the exact environment variables to set, and a list of known differences, are in the [package README](src/dual_arm_sync/README.md#reproducing-the-papers-experiments).

## Requirements

| Component | Notes |
|---|---|
| Ubuntu 22.04 + **ROS 2 Humble** | with Gazebo (`ros-humble-ros-gz`, `ros-humble-gazebo-ros-pkgs`) |
| Doosan ROS 2 packages | [`doosan-robot2`](https://github.com/doosan-robotics/doosan-robot2) (`humble` branch) in **`~/ros2_ws/src`**; provides `dsr_description2`, `dsr_gazebo2`, `dsr_control2`, `dsr_msgs2` |
| Python 3 | `numpy`, `scipy`, `python-fcl`, `trimesh`, `pycollada`, `matplotlib` (the real-mesh collision libraries are required by `step_251`-`step_256`) |

Two workspaces are used: **`~/ros2_ws`** holds the Doosan packages and **`~/dual_arm_ws`** (this repository) overlays it.

## Quick start

```bash
# 1. Doosan packages in ~/ros2_ws (follow the Doosan README for Gazebo and the emulator)
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone -b humble https://github.com/doosan-robotics/doosan-robot2.git
rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
cd ~/ros2_ws && colcon build && source install/setup.bash

# 2. This workspace, downloaded as dual_arm_ws
cd ~
git clone https://github.com/sriramreddysangu/dual_arm_ws.git dual_arm_ws
pip install numpy scipy matplotlib python-fcl trimesh pycollada

# 3. Build on top of ros2_ws
cd ~/dual_arm_ws
source ~/ros2_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash

# 4. Start Gazebo with two arms (terminal 1)
ros2 launch dual_arm_sync dual_gazebo.launch.py

# 5. Run the pipeline (terminal 2) from a scratch directory
source ~/ros2_ws/install/setup.bash && source ~/dual_arm_ws/install/setup.bash
mkdir -p ~/runs/dual && cd ~/runs/dual
ros2 run dual_arm_sync step_251    # enter targets for dsr01 / dsr02, IK + endpoint check
ros2 run dual_arm_sync step_252    # B-spline seed trajectories
ros2 run dual_arm_sync step_253    # real-mesh collision scan + timing prediction
ros2 run dual_arm_sync step_254    # Kuramoto phase synchronization
ros2 run dual_arm_sync step_255    # deformation + re-sync if needed
ros2 run dual_arm_sync step_256    # pre-flight and execution in Gazebo
```

> **Run from a scratch directory, in order.** Stages hand data to each other through JSON files (`s251_ik.json`, `s252_trajectories.json`, ...) that are read from and written to the *current working directory*.

`step_251` asks, for each arm, for a target as six joint angles (degrees) or an `X Y Z` position (metres, world frame), and for the duration. Prompts, options, outputs and non-interactive use are described in the [package README](src/dual_arm_sync/README.md#quick-start).

## Which scripts should I run?

Scripts are numbered by series. Within a series the last digit is the stage; series differ by number of arms and by how goals are chosen.

| Series | Arms | Purpose |
|---|---|---|
| `step_1` – `step_6` | 2 | Original dual-arm pipeline; `step_6` is a Gazebo benchmark |
| `step_7` – `step_10` | 2 | Interactive multi-waypoint variant |
| `step_11` – `step_16` | 4 | 4-arm pipeline (6 arm pairs); `step_16` is the benchmark |
| `step_21` – `step_27` | 6 | 6-arm pipeline, real-mesh scan, Kuramoto then retraction |
| `step_31` – `step_35`, `step_40` – `step_46` | 2 | Dual-arm temporal / "Lagrangian" variants with randomised benchmark (`step_46`) |
| `step_51` – `step_57` | 2 | Three-tier resolver (Kuramoto only -> home-pull -> deep home-pull) |
| `step_61` – `step_80` | 2 | SE(3) IK, surgical CP resolver, metrics, ablations and paper-figure generation |
| `step_81` – `step_87` | 6 (default; set `DUAL_ARM_CELL`) | Mesh-aware pipeline with closed-loop benchmark |
| `step_100` | 1 | Single-arm B-spline + Gazebo execution |
| `step_101` – `step_107` | 2 | Cell **dual** pipeline + benchmark |
| `step_111` – `step_117` | 4 | Cell **four** pipeline + benchmark |
| `step_121` – `step_127` | 6 | Cell **six** pipeline + benchmark |
| `step_131` – `step_137` | 9 | Cell **nine** pipeline + benchmark |
| `step_151` – `step_157` | 2 | Dual cell, manual or random-goal trials |
| `step_161` – `step_167` | 4 | Four cell, manual or random-goal trials |
| `step_171` – `step_177` | 6 | Six cell, manual or random-goal trials |
| `step_251` – `step_256` | 2 | Dual cell with manual goal entry (`step_251`, optional `--arms`, `--from-config` replay) |

The 100-series and later share the cell definitions in `_cell_bases.py` and the collision model in `_robot8x.py`; they are the most recent and are the recommended starting point. For the stage-by-stage description, data flow, configuration and environment variables, see the package README: **[`src/dual_arm_sync/README.md`](src/dual_arm_sync/README.md)**.

## Arm cells

Arms are named `dsr01` ... `dsr09` and sit on a grid with 1.39 m spacing in x and ±0.695 m in y (defined in `_cell_bases.py`):

| Cell | Arms |
|---|---|
| `dual` | dsr01, dsr02 |
| `four` | dsr01 – dsr04 |
| `six` | dsr01 – dsr06 |
| `nine` | dsr01 – dsr09 |

The earliest scripts (`step_1` – `step_16`) use a tighter layout with bases at y = ±0.5 m (x = 0 and 1.0 m).

## Author

Designed, implemented and maintained by **[@sriramreddysangu](https://github.com/sriramreddysangu)**, IIT Tirupati (me24s501@iittp.ac.in).
