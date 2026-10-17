# dual_arm_sync

Synchronized joint-space trajectory generation for **multiple Doosan M1013 arms** (2, 4, 6 or 9) operating in a shared workspace, with Gazebo execution and closed-loop benchmarks.

Reference implementation of **"Kuramoto Synchronized Configuration Space Trajectory Generation for an Ensemble of Manipulators Sharing a Common Workspace"** (ICRA 2027 submission, under review). Section and equation numbers below refer to that paper.

Closest prior work (different problem: workspace path constraints between robots): A. M. Kabir et al., *"Generation of synchronized configuration space trajectories of multi-robot systems"*, ICRA 2019, and its IJRR 2021 extension.

Author: [@sriramreddysangu](https://github.com/sriramreddysangu), IIT Tirupati.

Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Pipeline stages](#pipeline-stages)
5. [Paper-to-code map](#paper-to-code-map)
6. [Reproducing the paper's experiments](#reproducing-the-papers-experiments)
7. [Differences between the paper and this code](#differences-between-the-paper-and-this-code)
8. [Step-script index](#step-script-index)
9. [Data files](#data-files)
10. [Robot and cell model](#robot-and-cell-model)
11. [Configuration](#configuration)
12. [Launch files](#launch-files)
13. [Other modules](#other-modules)
14. [Testing](#testing)
15. [Troubleshooting](#troubleshooting)
16. [Known issues](#known-issues)

## Overview

Per-arm task-space goals are given and relative timing among the arms is free. For each arm the pipeline selects mutually clear goal configurations, fits an unmodified cubic B-spline path, scans all arm pairs for collision, and resolves interference **temporally first** (each arm advances along its path under a repulsive Kuramoto phase-oscillator model) and **geometrically only where timing is insufficient** (a perpendicular escape at the deepest contact, injected as B-spline control-point displacements, followed by a full re-run of the synchronization sweep). The result is a set of synchronized trajectories streamed at 100 Hz to each arm's Gazebo position controller.

```
 start/goal            B-spline             pairwise              Kuramoto
 (IK, endpoint  --->   trajectories  --->   collision scan  --->  timing resolver
  CSP)                 (cubic, 12 CP)       (mesh / capsule)           |
                                                                       | residual collisions?
                                                                       v
                                  Gazebo execution  <---  B-spline deformation (perpendicular escape) + Kuramoto re-sync
```

Key properties

- **Timing before geometry.** If staggering the arms in time removes every collision, the original paths are kept untouched (deviation D<sub>i</sub> = 0).
- **Real collision geometry.** The M1013 collision meshes are loaded as convex hulls and checked with FCL. If FCL/meshes are unavailable the code falls back to capsules and prints a loud warning (set `DUAL_ARM_REQUIRE_MESH=1` to abort instead).
- **Cell-parameterised.** The same code runs a `dual`, `four`, `six` or `nine` arm cell (`_cell_bases.py`).
- **Closed-loop benchmarks** plan randomised goals, execute them in Gazebo and record success categories.

## Installation

### Prerequisites

- Ubuntu 22.04 with **ROS 2 Humble** (`ros-humble-desktop`) and Gazebo
- Python 3 with `numpy` and `scipy`
- `python-fcl`, `trimesh`, `pycollada`: required for real-mesh collision. The `step_251`-`step_256` scripts abort without them (they set `DUAL_ARM_REQUIRE_MESH=1` by default)
- The Doosan ROS 2 packages (`dsr_description2`, `dsr_gazebo2`, `dsr_control2`, `dsr_msgs2`, ...) built in a separate workspace, `~/ros2_ws`

Two workspaces are used, with this one overlaying the other:

| Workspace | Contents |
|---|---|
| `~/ros2_ws` | Doosan Robotics packages (`doosan-robot2`, `humble` branch) |
| `~/dual_arm_ws` | This repository (package `dual_arm_sync`) |

### Steps

```bash
# 1. Doosan packages go in ~/ros2_ws (NOT inside dual_arm_ws)
sudo apt-get update
sudo apt-get install -y libpoco-dev libyaml-cpp-dev wget \
    ros-humble-control-msgs ros-humble-realtime-tools ros-humble-xacro \
    ros-humble-joint-state-publisher-gui ros-humble-ros2-control \
    ros-humble-ros2-controllers ros-humble-gazebo-msgs ros-humble-moveit-msgs \
    dbus-x11 ros-humble-moveit-configs-utils ros-humble-moveit-ros-move-group \
    ros-humble-gazebo-ros-pkgs ros-humble-ros-gz-sim ros-humble-ign-ros2-control \
    ros-humble-ros-gz

mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone -b humble https://github.com/doosan-robotics/doosan-robot2.git
rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y

# Doosan's emulator script (needs Docker), as in their instructions
cd ~/ros2_ws/src/doosan-robot2
chmod +x ./install_emulator.sh && sudo ./install_emulator.sh

cd ~/ros2_ws && colcon build
source ~/ros2_ws/install/setup.bash

# 2. This workspace, downloaded as ~/dual_arm_ws
cd ~
git clone https://github.com/sriramreddysangu/dual_arm_ws.git dual_arm_ws

# 3. Python dependencies
pip install numpy scipy matplotlib python-fcl trimesh pycollada

# 4. Build dual_arm_ws on top of ros2_ws
cd ~/dual_arm_ws
source ~/ros2_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

For Gazebo (Ignition) installation and the emulator, follow the [Doosan Robotics `humble` README](https://github.com/doosan-robotics/doosan-robot2/tree/humble). To avoid sourcing by hand in every terminal, add both lines to `~/.bashrc`:

```bash
source ~/ros2_ws/install/setup.bash
source ~/dual_arm_ws/install/setup.bash
```

### Collision meshes

`mesh_collision.py` needs `m1013_parts.npz` (convex hulls of the ten M1013 collision parts). It is built automatically on first use: the `.dae` files are taken from a local `dsr_description2` checkout if one is found (searched under `$DUAL_ARM_ROS2_WS`, the package directory, `~/ros2_ws` and `~`), otherwise downloaded from the DoosanRobotics `doosan-robot2` repository. To build or rebuild explicitly and run the self-test:

```bash
python3 dual_arm_sync/mesh_collision.py --build /path/to/ros2_ws
```

Set `DUAL_ARM_NO_AUTOBUILD=1` to disable the automatic build, or `DUAL_ARM_MESH_NPZ=/path/to/file.npz` to use a different file.

## Quick start

The recommended entry point is the **251 -> 256** series: you enter one target per arm, and the six stages plan, resolve and execute the motion for the two-arm cell (`dsr01`, `dsr02`).

### 1. Start Gazebo (terminal 1)

```bash
source ~/ros2_ws/install/setup.bash && source ~/dual_arm_ws/install/setup.bash
ros2 launch dual_arm_sync dual_gazebo.launch.py          # 2 arms: dsr01 (y=+0.695), dsr02 (y=-0.695)
```

Wait until both arms have spawned and their controllers are up.

### 2. Run step_251 to step_256 (terminal 2)

Run the steps **in order, from the same scratch directory**: each step reads the JSON written by the previous one from the current directory.

```bash
source ~/ros2_ws/install/setup.bash && source ~/dual_arm_ws/install/setup.bash
mkdir -p ~/runs/dual && cd ~/runs/dual

ros2 run dual_arm_sync step_251    # enter targets, IK + endpoint check
ros2 run dual_arm_sync step_252    # B-spline seed trajectories
ros2 run dual_arm_sync step_253    # real-mesh collision scan + timing-feasibility prediction
ros2 run dual_arm_sync step_254    # Kuramoto phase synchronization (timing only)
ros2 run dual_arm_sync step_255    # deformation + re-sync if needed (pass-through otherwise)
ros2 run dual_arm_sync step_256    # pre-flight, then execute in Gazebo
```

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `step_251` | Reads each arm's live START pose from Gazebo, asks for each arm's TARGET and the duration, solves IK, and checks that the two target poses are mutually clear on the real mesh. Stops if they collide (`--force` overrides) | Gazebo joint states | `s251_ik.json`, `s251_config.json` |
| `step_252` | Clamped cubic B-spline seed per arm (12 control points), duration scaled to joint velocity / acceleration limits, all arms on one clock | `s251_ik.json` | `s252_trajectories.json` |
| `step_253` | Pairwise scan on the real collision mesh. For each colliding pair it grids the coordination domain U<sub>ij</sub> and predicts whether timing alone suffices (`KURAMOTO_ALONE`, with the lag lambda and the blocked fraction of U<sub>ij</sub>) or deformation is needed (`NEEDS_DEFORMATION`) | `s252_trajectories.json` | `s253_collision_map.json` |
| `step_254` | Kuramoto phase synchronization along the unmodified paths; reports the chosen configuration, minimum clearance, new duration, simultaneity and brake events. Compares its result with the step_253 prediction | `s252_trajectories.json`, `s253_collision_map.json` | `s254_resolved.json` |
| `step_255` | If step_254 left residual pairs: perpendicular-escape deformation of the contacting arc followed by a full re-sync. If not, passes the plan through unchanged | `s254_resolved.json` (+ `s252`, `s253`) | `s255_synchronized.json` |
| `step_256` | Refuses to run unless step_255 verified the plan collision-free; pre-flight check on the exact mesh; ramps the arms to the planned start; streams joint commands; verifies end poses | `s255_synchronized.json` (+ `s252`, `s253`, `s251_config.json`) | `s256_execution.json` |

### 3. What you enter in step_251

For each arm you are shown its current Gazebo pose and asked for a target:

```
[dsr01] TARGET  6 joints j1..j6 (deg)  OR  3 numbers X Y Z (m):
```

- **Six numbers**: joint angles j1..j6 in degrees, used as typed (clamped to the joint limits with a warning).
- **Three numbers**: a target in world coordinates in metres; IK is solved for it. The arm bases are at y = +0.695 m (`dsr01`) and y = -0.695 m (`dsr02`), z = 0.
- Then `Duration [s] (Enter=10.0):`.

Non-interactive form (no prompts):

```bash
DUAL_ARM_S251_MODE=cartesian \
DUAL_ARM_DSR01_GOAL=0.35,0.40,0.70 \
DUAL_ARM_DSR02_GOAL=0.35,-0.40,0.70 \
ros2 run dual_arm_sync step_251 --duration 10
```

`DUAL_ARM_S251_MODE` is `joint` by default (goals are then six angles in degrees) or `cartesian` (goals are `x,y,z` in metres). `DUAL_ARM_<ARM>_START` (six degrees) overrides the live start pose.

Options of `step_251`:

| Option | Meaning |
|---|---|
| `--arms dsr01,dsr02` | Arms to include (this series is pinned to the two-arm cell) |
| `--from-config s251_config.json` | Replay a saved configuration exactly (saved START poses are reused) |
| `--start-live` | With `--from-config`, take START from Gazebo instead of the saved values |
| `--start-home` | Use all-zero joints as START on purpose (otherwise a missing START pose aborts, to avoid the arm snapping across the cell) |
| `--duration SECONDS` | Duration without prompting |
| `--force` | Write the configuration even if the target poses collide |

Replay a run later (for example for a video) with `ros2 run dual_arm_sync step_251 --from-config s251_config.json`, then `step_252` ... `step_256` again.

### 4. Reading the output

- `step_253` prints a **PAIRWISE RESULT** table: minimum distance, collision percentage, and which mechanism the pair needs.
- `step_254` ends with `KURAMOTO SUCCESS (timing alone)` or `timing cannot separate ... deferring to step_255`.
- `step_255` ends with the next-step hint, or `UNRESOLVED -- step_256 pre-flight will refuse to execute this`. In that case retype a target in `step_251`, or raise `DUAL_ARM_TIME_BUDGET_S`.
- `step_256` saves `s256_execution.json` (per-arm joint and end-effector errors, settling, method, duration overhead).

Settings for these steps: `DUAL_ARM_S251_WAIT_S` (20, seconds to wait for joint states), `DUAL_ARM_S253_GRID` (60) and `DUAL_ARM_S253_LAGSCAN` (set to `0` to skip the timing-feasibility grid), `DUAL_ARM_S256_APPROACH` (set to `0` to skip the ramp to the planned start), `DUAL_ARM_S256_APPROACH_MIN_S` (1.5), `DUAL_ARM_CARTESIAN_SAMPLES` (60).

### 5. Other ways to run

| Goal | Command |
|---|---|
| Random or scripted goals, stage by stage, dual cell | `step_151` ... `step_156` (see below) |
| Closed-loop randomised benchmark, dual / four / six arms | `step_157` / `step_167` / `step_177` (`DUAL_ARM_TRIALS`, `DUAL_ARM_SEED`) |
| The paper's Table III experiments | `step_107` / `step_117` / `step_127` ([how](#reproducing-the-papers-experiments)) |

For `step_151` / `161` / `171`, goals are chosen as follows:

| Mode | How |
|---|---|
| Interactive | Run in a terminal; you are prompted for goals and duration. |
| Random | Not a TTY, or `DUAL_ARM_S51_AUTO=1`; goals are sampled with `DUAL_ARM_SEED`. |
| Scripted | `DUAL_ARM_S51_MODE=cartesian` (default) or `joint`, then one `DUAL_ARM_<ARM>_GOAL` per arm: `x,y,z` in metres, or six angles in **degrees** in joint mode. `DUAL_ARM_<ARM>_START` (six degrees) is optional. |

## Pipeline stages

The 100-series and later all follow the same six stages. Replace `N` with the series base (`10`, `11`, `12`, `13`, `15`, `16`, `17`, and `25`):

| Stage | Script | What it does | Reads | Writes |
|---|---|---|---|---|
| 1 | `step_N1` | Inverse kinematics for each arm; picks a consistent set of endpoint IK solutions that are mesh-clear (endpoint CSP) | Gazebo joint states | `sN1_ik.json` |
| 2 | `step_N2` | Cubic B-spline (degree 3, 12 control points) from start to goal per arm, scaled to the requested duration | `sN1_ik.json` | `sN2_trajectories.json` |
| 3 | `step_N3` | Pairwise collision scan along the trajectories with surface clearance (default 0.05 m) and deepest-link tracking | `sN2_trajectories.json` | `sN3_collision_map.json` |
| 4 | `step_N4` | **Kuramoto phase synchronization** (paper Sec. IV-E). Arms advance along their unmodified paths with a multiplicative, repulsive phase-rate law; searches the lag bound lambda in ascending order and per-pair leader orderings. If it clears every pair, reports success; otherwise passes the residual pairs on | trajectories, collision map | `sN4_resolved.json` |
| 5 | `step_N5` | **B-spline deformation (perpendicular escape)** (paper Sec. IV-F; called "retraction" in file names and JSON keys) for pairs timing could not separate, then Stage 4 is re-invoked on the modified paths. Pass-through when stage 4 was sufficient. See [differences](#differences-between-the-paper-and-this-code) for fallbacks beyond the paper | trajectories, `sN4_resolved.json` | `sN5_synchronized.json` |
| 6 | `step_N6` | ROS 2 executor: publishes joint commands to each arm's controller and monitors tracking | `sN5_synchronized.json` | `sN6_execution.json` |
| 7 | `step_N7` | Closed-loop benchmark: runs stages 1-6 for many random trials and summarises outcomes | all of the above | `<prefix>_results.json/.csv` |

For the 100-series the "N" prefix is `10`/`11`/`12`/`13` (stages `101`-`107`, `111`-`117`, ...), and the file names use the stage numbers (`s101_ik.json`, `s102_trajectories.json`, `s103_collision_map.json`, `s104_resolved.json`, `s105_synchronized.json`, `s106_execution.json`). In the 150/160/170 series the stage 7 script is the benchmark and stages 1-6 are `151`-`156` etc.

`step_101`, like the other stage-1 modules in its series, is a **library module**: running it directly just prints a notice. Use `step_107` (or the stage-by-stage 151-156 flow) instead.

Benchmark outcome categories (`_bench_common.py`): `NO_COLLISION`, `KURAMOTO` (resolved by timing alone; paper Table III "Resolved by Kuramoto alone"), `RETRACTION` (needed geometric deformation then re-sync; paper "Resolved by deformation+Kuramoto"), `UNRESOLVED`, `IK_FAIL` (the 150-series harnesses also emit `START_COLLIDING`).

## Paper-to-code map

| Paper | Content | Code (100-series names; N1..N7 = stages 1..7 of each series) |
|---|---|---|
| Sec. III, Eq. (1), (5) | Inter-arm separation d<sub>ij</sub> on the true collision mesh; clearance gate epsilon | `mesh_collision.py` (`pair_min_dist`, `pair_collides`, FCL on 10 convex parts per arm); `CLEAR_M = DUAL_ARM_CLEARANCE_M` (0.05 m) |
| Sec. IV-B, Eq. (9) | Stage 1: multi-seed SQP IK, rank by (operation time, weighted joint travel) | `step_101.py`: `_ik_seeds`, `solve_ik` (SLSQP, keeps up to 40 solutions, 10 mm position tolerance), `score`, `JOINT_COST_W` |
| Sec. IV-B, Eq. (2) | Min-conflicts search for a mutually clear goal assignment at epsilon + delta, then epsilon | `step_101.py`: `repair_endpoint_collisions` (margins `CLEAR_M + 0.02`, then `CLEAR_M`; 8 restarts); infeasible -> `_endpoint_ok = False` |
| Sec. IV-C, Eq. (10) | Stage 2: clamped cubic B-spline, second / second-to-last control points duplicated onto the endpoints, duration scaled to velocity / acceleration limits, common clock | `step_102.py`: `seed_cp` (12 control points), `make_knots`, `scale_duration` (x1.05 margin), resample to the longest duration |
| Sec. IV-D, Eq. (11) | Stage 3: scan all N(N-1)/2 pairs; bounding-sphere lower bound skips the exact query | `step_103.py`; `_robot8x.mesh_scan` (stride `DUAL_ARM_SCAN_STRIDE` = 2); `_caps_surface` -> `mesh_collision.broad_lower_bound` (skip if bound > `PREFILTER_M`) |
| Sec. IV-E, Eq. (13) | Multiplicative rate law, phi-dot = omega * s, omega = 1/T0 | `step_104.py` `kuramoto()`: `omega = 1/duration`, `dphi = omega * scale` |
| Sec. IV-E, Eq. (14) | Brake factor from d<sub>safe</sub> / d<sub>hard</sub>, applied to the trailing arm of flagged pairs only | `kuramoto()`: `d_hard = CLEAR_M`, `d_safe = CLEAR_M + band`; `t = a if phi[a] < phi[b] else b`; coupling restricted to `conflict` pairs |
| Sec. IV-E, Eq. (15) | Lag floor phi >= min(1, t/T0 - lambda) | `kuramoto()`: `phi = max(phi, min(1, nominal - lag_cap))`; lambda candidates `DUAL_ARM_LAG_CAPS` (0.05 ... 1.00), tried ascending in `kuramoto_sweep` |
| Sec. IV-E, 5) | Leader orderings, each verified on the exact mesh before acceptance | `lead_combos` (hold the intended follower for 5% of T0), `kuramoto_sweep` (acceptance via `mesh_scan`) |
| Sec. IV-F, Eq. (16)-(17) | Two-sided escape dq<sub>i</sub> = +alpha J<sub>i</sub>^T n, dq<sub>j</sub> = -alpha J<sub>j</sub>^T n at the witness points | `_robot8x.escape_dq` / `escape_dq_mesh`, `witness_pair_mesh`, `mesh_collision.mesh_witness_pair`, `point_jacobian_urdf` |
| Sec. IV-F, Eq. (18) | Minimum-norm control-point update, endpoints excluded | `step_105.py` `apply_curve_delta` (first and last basis weights zeroed) |
| Sec. IV-F | Local refinement before the push; step halving until the global violation count strictly decreases; pairs sharing an arm repaired jointly; Stage 4 re-invoked after each accepted change | `refine_cp`; `escape_repair` (`LS_STEPS` = 5, alpha0 = 0.15, `MAX_REPAIR` = 25); `clusters_of`; round loop in `step_105.main()` |
| Sec. IV-G, Eq. (19) | Pre-flight on exact geometry; settle window t = tau ln(e0 / e_tol), e0 ~ v_peak tau | `step_106.py`: `preflight`, `settle_budget` (`DUAL_ARM_CTRL_TAU` = 5.3 s) |
| Sec. V-A, Table II | Cells of 2 / 4 / 6 arms on a 1.39 m grid | `_cell_bases.py` (`dual`, `four`, `six`); harnesses `step_107` / `step_117` / `step_127` |
| Sec. V-B, Table III | Outcome by ensemble size, n = 150 per cell | `step_107` / `117` / `127` write `s107_config_results.json/.csv` etc.; categories from `_bench_common.py` |
| Sec. V-C, Table IV | Two-sided vs one-sided escape on a 1.62 cm contact | `escape_dq(..., both=True/False)`; the 2.35 deg vs 5.15 deg figures are recorded in the `escape_dq_mesh` docstring. No standalone script reproduces the table |
| Figs. 1, 5 | Six-arm cell, 2/4/6-arm cells in the mesh viewer | `_cell_bases.py`; `dual_gazebo`, `multi_arm_gazebo`, `multiple_arm_gazebo` launch files |
| Figs. 3, 4 | Coordination domain / phase trajectories; deformation before and after refinement | No dedicated script identified in this repository |

## Reproducing the paper's experiments

Table II values and what to set. Defaults in the code differ for four entries (marked **set**):

| Table II item | Paper | Code default | How to match |
|---|---|---|---|
| Manipulators | M1013, 6 DOF | same | - |
| Ensemble sizes | 2, 4, 6 | `step_107`, `step_117`, `step_127` | - |
| Base spacing | 1.39 m | 1.39 m (`_cell_bases.py`) | - |
| Collision geometry | 10 convex parts per arm | same | - |
| Clearance gate epsilon | 0.05 m | 0.05 m | - |
| Nominal duration T0 | 10 s | 10 s | - |
| Command rate | 100 Hz | 100 Hz | - |
| End-effector tolerance | 10 mm | 50 mm (benchmark), 25 mm (executor) | **set** `DUAL_ARM_EE_TOL_MM=10` |
| Stage 5 iteration cap | 25 per cluster | 25 | - |
| Stage 5 time budget | 300 s | 600 s | **set** `DUAL_ARM_TIME_BUDGET_S=300` |
| Minimum brake rate s<sub>min</sub> | 0.22 | 0.0 | **set** `DUAL_ARM_CREEP_MIN=0.22` (see difference 1 below) |
| Trials per cell | 150 | 20 | **set** `DUAL_ARM_TRIALS=150` |

```bash
export DUAL_ARM_TRIALS=150 DUAL_ARM_SEED=0
export DUAL_ARM_EE_TOL_MM=10 DUAL_ARM_TIME_BUDGET_S=300 DUAL_ARM_CREEP_MIN=0.22
export DUAL_ARM_CLEARANCE_M=0.05 DUAL_ARM_REQUIRE_MESH=1     # abort instead of falling back to capsules

# terminal 1: Gazebo for the cell (2, 4 or 6 arms)
ros2 launch dual_arm_sync dual_gazebo.launch.py              # 2 arms
# ros2 launch dual_arm_sync multi_arm_gazebo.launch.py       # 4 arms
# ros2 launch dual_arm_sync multiple_arm_gazebo.launch.py    # 6 arms

# terminal 2: one run directory per cell
mkdir -p ~/runs/paper_2arm && cd ~/runs/paper_2arm
ros2 run dual_arm_sync step_107      # 4 arms: step_117   6 arms: step_127
```

Results are written to the run directory (`s107_config_results.json` and `.csv` for the dual cell). Execution times in Table III are measured in Gazebo, so run with Gazebo live and `DUAL_ARM_EXEC` left at its default of `1`; with `DUAL_ARM_EXEC=0` only planning is evaluated.

## Differences between the paper and this code

These were found by reading the code against the paper. Items 1-6 can change reported numbers or the description of a mechanism and should be reconciled (by changing the code or the paper) before relying on either.

1. **Minimum brake rate.** Paper (Eq. 14, Table II): s<sub>min</sub> = 0.22, "never a full stop". Code: `CREEP_MIN` defaults to 0.0, and `kuramoto()` sets the factor to exactly 0 whenever d < d<sub>hard</sub> regardless of `CREEP_MIN`. A full stop is therefore possible; termination is still guaranteed by the lag floor (Eq. 15). Matching the paper needs `DUAL_ARM_CREEP_MIN=0.22` and removing the `0.0 if d < d_hard` branch in `kuramoto()` (in `step_84`, `104`, `114`, `124`, `134`, `154`, `164` and `174`; `step_254` reuses the stage 4 code).
2. **Stage 5 time budget.** Paper: 300 s. Code: `DUAL_ARM_TIME_BUDGET_S` defaults to 600 s. (Stage 4 has its own 300 s budget, `DUAL_ARM_S24_BUDGET_S`.)
3. **Local refinement is not literal knot insertion.** The paper describes insertion near u* so the edit stays confined to the contacting arc (Fig. 4: 53% -> 23% of the path). `refine_cp` instead adds 4 control points (up to 28) and refits all of them by least squares on a uniformly re-spaced knot vector; its `u` argument is not used. Curve geometry is preserved only approximately and the refinement is global, not local to u*. Locality then comes from the smaller basis support.
4. **Fallbacks in Stage 5 that the paper does not describe.** After two rounds the code also (a) clears residual pairs against the other arm's parked start/goal pose with a one-sided escape (`stage_C`, `escape_static`) and (b) as a last resort builds a *topological sequential schedule* in K phases (`topo_sequential`, makespan K x T0). All of these set `method = kuramoto_plus_witness_escape`, so benchmark rows are labelled `RETRACTION` and counted under "Resolved by deformation+Kuramoto", including sequential schedules. This affects Table III's resolution split and the mean execution time and overhead of interfering trials. Either disclose these fallbacks or disable them before running the paper experiments.
5. **Goal sampling in the benchmark.** Paper: goals drawn uniformly by volume from the intersection W<sub>N</sub> of all arms' reachable workspaces, with no reachability pre-filter. Code (`step_107.sample_goal`): a box (|x| <= 1.25 m, |y| <= 0.75 m, 0.35 <= z <= 1.15 m) with rejection to 0.35 < r < 1.25 m from *that arm's* shoulder, independently per arm, and no intersection across arms.
6. **Failure categories.** Paper: `IK_FAIL`, `ENDPOINT_CSP_FAIL`, `STAGE_ERROR`. Code: endpoint-CSP failures are reported as `IK_FAIL` with reason "no mesh-clear endpoint assignment" (and `failed_stage = step_101_endpoint_infeasible`); the 150-series adds `START_COLLIDING`. Use the `reason` / `failed_stage` fields to separate them.
7. **Leader orderings.** Paper: all 2^K assignments per candidate lambda. Code (`lead_combos`): only the first 4 interfering pairs are enumerated (at most 16 orderings, `DUAL_ARM_LEAD_COMBOS`), each realised by holding the follower for 5% of T0. The lambda search runs over the discrete grid `DUAL_ARM_LAG_CAPS` (0.05, 0.10, 0.15, 0.22, 0.30, 0.45, 0.70, 1.00).
8. **Brake band.** d<sub>safe</sub> = epsilon + band with band in {0.11, 0.09, 0.16} m, tried as three configurations per lambda; the paper gives no numeric d<sub>safe</sub>. The first element of each `BRAKE_CFGS` tuple (`k_lag`) is passed to `kuramoto()` but not used. Braking decisions use capsule distances; acceptance uses the exact mesh gate, consistent with the paper's statement that Eq. (14) never certifies a result.
9. **End-effector tolerance.** The IK position tolerance is 10 mm (`IK_TOL_POS`), but the execution-success tolerance is 25 mm (executor) or 50 mm (benchmark) unless `DUAL_ARM_EE_TOL_MM` is set.
10. **Settling window.** Code uses t = tau ln(e0 / e_tol) + 2 s, clamped to [8, 45] s, with tau = 5.3 s by default (the paper does not state tau).
11. **Scan sampling and prefilter.** Scans visit every second sample (`DUAL_ARM_SCAN_STRIDE` = 2, i.e. 50 Hz) and skip the exact query when the sphere lower bound exceeds `DUAL_ARM_PREFILTER_M` (default 0.05 m), a variable separate from epsilon. Keep `PREFILTER_M >= CLEARANCE_M` if you change epsilon. The paper's Eq. (11) skip test uses epsilon.
12. **Witness fallback.** When meshes interpenetrate, the escape direction falls back to the capsule centre-line witness (the paper calls this the medial-axis fallback).

The nine-arm cell (`step_131`-`137`) goes beyond the paper, which evaluates 2, 4 and 6 arms.

## Step-script index

| Scripts | Cell | Notes |
|---|---|---|
| `step_1`-`step_5` | 2 arms (y = ±0.5 m) | Original pipeline: IK, B-spline, collision check, Kuramoto + alternating IK + B-spline refinement, synchronized execution |
| `step_6` | 2 arms | Gazebo benchmark: `--trials` (default 100), `--seed` (0), `--duration` (10 s) |
| `step_7`-`step_10` | 2 arms | Interactive multi-waypoint B-spline variant |
| `step_11`-`step_15` | 4 arms | 4-arm version of 1-5 (all 6 arm pairs checked) |
| `step_16` | 4 arms | 4-arm benchmark (`--trials`, `--seed`, `--duration`) |
| `step_21`-`step_26`, `step_27` | 6 arms | Joint-aware IK selection, global B-spline seed, real-mesh scan, Kuramoto-alone resolver, retraction + Kuramoto, executor; `step_27` is a closed-loop benchmark |
| `step_31`-`step_35` | 2 arms | Greedy-pairwise IK, path-velocity temporal resolver, executor |
| `step_40`-`step_45`, `step_46` | 2 arms | Same flow with deepest-link tracking and a Kuramoto-alone step; `step_46` is a randomised benchmark |
| `step_51`-`step_57` | 2 arms | Three-tier resolver: Kuramoto-only, Kuramoto-centric home-pull, deep home-pull; `step_57` executes and summarises tiers |
| `step_61`-`step_67` | 2 arms | SE(3) IK, B-spline, scanner, surgical CP resolver, Kuramoto synchronizer, executor, metrics collector |
| `step_68`-`step_80` | 2 arms | Benchmark, baselines, quality metrics, retraction / Kuramoto / lemma checks, experiment driver (`step_79`: `--trials`, `--seed`, `--outdir`), paper-figure generator (`step_80`) |
| `step_81`-`step_87` | `DUAL_ARM_CELL` (default `six`) | Mesh-aware pipeline and closed-loop benchmark |
| `step_100` | 1 arm | Single-arm B-spline + Gazebo execution |
| `step_101`-`step_107` | `dual` | Staged pipeline + benchmark |
| `step_111`-`step_117` | `four` | Staged pipeline + benchmark |
| `step_121`-`step_127` | `six` | Staged pipeline + benchmark |
| `step_131`-`step_137` | `nine` | Staged pipeline + benchmark |
| `step_151`-`step_157` | `dual` | Manual / random goals, stage by stage, plus benchmark |
| `step_161`-`step_167` | `four` | As above |
| `step_171`-`step_177` | `six` | As above |
| `step_251`-`step_256` | `dual` | Manual configuration entry with replay (`--from-config`) |
| `step_plan`, `step_34_diag`, `step_64_viz` | - | Planning helper, diagnostic, visualization |

## Data files

Stages communicate through JSON files in the working directory. For the dual-arm 150-series:

```
s151_ik.json  ->  s152_trajectories.json  ->  s153_collision_map.json
                                                       |
                         s154_resolved.json  <---------+
                                |
                s155_synchronized.json  ->  s156_execution.json
```

The earliest scripts (1-16) use unprefixed names: `ik_solutions.json` -> `trajectories.json` -> `collision_report.json` -> `synchronized_trajectories.json` -> `execution_report.json`. The 21-87 series use `sNN_*.json` names (for example `s21_ik.json`, `s25_synchronized.json`).

Trajectory files contain, per arm, `trajectory.positions` (joint angles at 100 Hz), and a top-level `arm_names` list.

## Robot and cell model

### Kinematics (modified DH, M1013)

Link constants: L1 = 0.1525 m, L2 = 0.620 m, L3 = 0.559 m, L4 = 0.121 m, A = 0.0345 m.

| Joint | alpha | a | theta offset | d |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | L1 |
| 2 | -pi/2 | 0 | -pi/2 | A |
| 3 | 0 | L2 | pi/2 | 0 |
| 4 | pi/2 | 0 | 0 | L3 |
| 5 | -pi/2 | 0 | 0 | 0 |
| 6 | pi/2 | 0 | 0 | L4 |

Joint limits (rad): j1 ±2pi, j2 ±1.6493, j3 ±2.7925, j4-j6 ±2pi. Velocity limits (rad/s): 2.094, 2.094, 3.140, 3.927, 3.927, 3.927. Acceleration limits (rad/s²): 8, 8, 8, 12, 12, 12. Command rate: 100 Hz.

The DH joint order differs from the controller order; `DH_TO_CMD = [0, 1, 3, 4, 2, 5]` maps between them. Executors publish to `/{arm}/gz/dsr_position_controller/commands` and listen on `/{arm}/gz/joint_states` (or `/{arm}/joint_states`).

### Collision model

- Primary: FCL on convex hulls of the ten `MF1013_*` collision parts (`mesh_collision.py`).
- Fallback: capsule model (link radii 0.10, 0.10, 0.08, 0.08, 0.06, 0.06 m; `SAFETY_MARGIN` 0.12 m), with a warning.
- Clearance threshold for the scan: `DUAL_ARM_CLEARANCE_M` (default 0.05 m).

### Cells (`_cell_bases.py`)

Base positions in metres (z = 0):

| Arm | dual | four | six | nine |
|---|---|---|---|---|
| dsr01 | (0, 0.695) | (0, 0.695) | (0, 0.695) | (0, 0.695) |
| dsr02 | (0, -0.695) | (0, -0.695) | (0, -0.695) | (0, -0.695) |
| dsr03 | | (1.39, 0.695) | (1.39, 0.695) | (1.39, 0.695) |
| dsr04 | | (1.39, -0.695) | (1.39, -0.695) | (1.39, -0.695) |
| dsr05 | | | (-1.39, 0.695) | (-1.39, 0.695) |
| dsr06 | | | (-1.39, -0.695) | (-1.39, -0.695) |
| dsr07 | | | | (0, 0) |
| dsr08 | | | | (1.39, 0) |
| dsr09 | | | | (-1.39, 0) |

Most step scripts pin their cell at import time (`os.environ['DUAL_ARM_CELL'] = ...`). The cell is chosen by `DUAL_ARM_CELL` only for scripts that do not pin it (default `six`). The oldest scripts (`step_1`-`step_16`) use their own layout with bases at y = ±0.5 m and x = 0 / 1.0 m.

## Configuration

### `config/pipeline_params.yaml`

Parameters for the original ROS-node-style pipeline (collision threshold, RRT settings, B-spline degree, joint / velocity / acceleration / jerk limits, workspace bounds, position-controller gains). Excerpt:

```yaml
dual_arm_pipeline:
  ros__parameters:
    collision_threshold: 0.15
    hierarchical_max_depth: 5
    samples_per_segment: 20
    rrt_max_iterations: 5000
    bspline_degree: 3
    trajectory_duration: 10.0
    control_rate: 100.0
    max_ik_solutions: 8
```

`config/sample_workspace_config.yaml` holds example workspace bounds, base positions and start / target zones per arm.

> The step scripts above are configured through command-line options and the environment variables below, not through this YAML file.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DUAL_ARM_CELL` | `six` | Cell for scripts that do not pin it: `dual`, `four`, `six`, `nine` |
| `DUAL_ARM_CLEARANCE_M` | `0.05` | Minimum allowed surface clearance between arms (m) |
| `DUAL_ARM_REQUIRE_MESH` | unset | If set, abort when real-mesh collision is unavailable |
| `DUAL_ARM_MESH_NPZ` | `dual_arm_sync/m1013_parts.npz` | Path to the mesh-hull file |
| `DUAL_ARM_NO_AUTOBUILD` | unset | Do not auto-build the hull file |
| `DUAL_ARM_ROS2_WS` | unset | Extra location to search for `dsr_description2` meshes |
| `DUAL_ARM_DURATION` | `10.0` (151/161/171) | Trajectory duration (s) |
| `DUAL_ARM_SEED` | `0` | Random seed for goal sampling |
| `DUAL_ARM_TRIALS` | `20` | Trials per benchmark run |
| `DUAL_ARM_EXEC` | `1` | `0` = plan only, do not execute in Gazebo |
| `DUAL_ARM_RECOVER` | `1` | Enable recovery between trials in benchmarks |
| `DUAL_ARM_EE_TOL_MM` | `25` (executor) / `50` (benchmark) | End-effector tolerance for success |
| `DUAL_ARM_S51_MODE` | `cartesian` | Scripted goal mode: `cartesian` or `joint` |
| `DUAL_ARM_S51_AUTO` | unset | `1` forces random goals even on a TTY |
| `DUAL_ARM_S251_MODE` | `joint` | Scripted goal mode for `step_251`: `joint` (degrees) or `cartesian` (metres) |
| `DUAL_ARM_S251_WAIT_S` | `20` | Seconds `step_251` waits for Gazebo joint states |
| `DUAL_ARM_S253_GRID` / `_LAGSCAN` | `60` / `1` | Resolution of the U<sub>ij</sub> timing-feasibility grid / set `0` to skip it |
| `DUAL_ARM_S256_APPROACH` / `_APPROACH_MIN_S` | `1` / `1.5` | Ramp the arms to the planned start before executing (`0` = skip) / minimum ramp time (s) |
| `DUAL_ARM_CARTESIAN_SAMPLES` | `60` | Samples used for the Cartesian report in `step_256` |
| `DUAL_ARM_<ARM>_GOAL` / `_START` | - | Per-arm goal / start for scripted runs (e.g. `DUAL_ARM_DSR01_GOAL`) |
| `DUAL_ARM_MAX_LAG` | `1.0` | Largest lag bound lambda considered |
| `DUAL_ARM_LEAD_COMBOS` | `16` | Cap on leader orderings tried per lambda (only the first 4 interfering pairs are enumerated) |
| `DUAL_ARM_LAG_CAPS` | `0.05,0.10,0.15,0.22,0.30,0.45,0.70,1.00` | Candidate lag bounds lambda (Eq. 15), tried ascending |
| `DUAL_ARM_CREEP_MIN` | `0.0` | Minimum brake factor s<sub>min</sub> (paper: 0.22) |
| `DUAL_ARM_PREFILTER_M` | `0.05` | Skip exact mesh query when the sphere lower bound exceeds this (m) |
| `DUAL_ARM_SCAN_STRIDE` | `2` | Sample stride used by collision scans |
| `DUAL_ARM_S24_BUDGET_S` | `300` | Time budget of the Stage 4 Kuramoto sweep (s) |
| `DUAL_ARM_TIME_BUDGET_S` | `600` | Time budget of Stage 5 (paper: 300 s) |
| `DUAL_ARM_NOPROGRESS_ROUNDS` | `2` | Stop Stage 5 after this many rounds without progress (also capped at 8 rounds) |
| `DUAL_ARM_ESCAPE_ALPHA` / `_ITER` | `0.15` / `25` | Escape step size / max repair iterations |
| `DUAL_ARM_LINE_SEARCH` | `5` | Line-search steps in escape moves |
| `DUAL_ARM_REFINE_CP` / `_REFINE_MAX` | `4` / `28` | Control-point refinement increment / maximum |
| `DUAL_ARM_SETTLE_S`, `_SETTLE_MIN`, `_SETTLE_MAX`, `_CTRL_TAU` | `0`, `8`, `45`, `5.3` | Executor settling-time behaviour (s) |

Further tuning variables exist in the scripts; search for `os.environ.get('DUAL_ARM_` for the full list.

## Launch files

| File | Purpose |
|---|---|
| `dual_gazebo.launch.py` | Gazebo with 2 M1013 arms (`dsr01` white at y = +0.695, `dsr02` blue at y = -0.695); arguments `name1/2`, `color1/2`, `model1/2`, `x1..`, `y1..`, ... |
| `multi_arm_gazebo.launch.py` | Gazebo with 4 arms |
| `multiple_arm_gazebo.launch.py` | Gazebo with 6 arms |
| `multiple_arms_gazebo.launch.py` | Gazebo with 7 arms |
| `dual_rviz.launch.py` | RViz visualisation of two arms (default y = ±0.5) |
| `dual_moveit.launch.py` | Dual-arm MoveIt setup |

Gazebo is started once and the robots are spawned through `dsr_spawn_on_gazebo.launch.py`.

## Other modules

Shared helpers

| Module | Role |
|---|---|
| `_robot.py`, `_robot2x.py`, `_robot4x.py`, `_robot5x.py`, `_robot6x.py`, `_robot8x.py` | Kinematics, link geometry and collision functions for each generation of the pipeline; `_robot8x` is the current one (mesh + capsule, witness points, escape directions) |
| `_cell_bases.py` | Cell definitions and `select_cell()` |
| `_bench_common.py`, `bench_core.py` | Benchmark helpers: goal sampling, all-pair scans, outcome classification |
| `constants.py` | DH table, link lengths, joint limits as classes |
| `mesh_collision.py`, `mesh_collision_checker.py` | FCL mesh collision and checker |

Earlier standalone modules (kept for reference; the numbered step scripts do not depend on them): `collision_checker`, `dual_arm_ik_solver`, `ik_solver`, `ik_interactive`, `trajectory_generation`, `hierarchical_collision_resolver`, `rrt_connect_planner`, `local_deformation`, `kuramoto_sync`, `kuramoto_synchronization`, `gazebo_executor`, `multi_arm_core`, `multi_arm_trajectory_planner`, `run_pipeline`, `enhanced_testing_orchestrator`, `analyze_timing`, and the `quad_*` family for four arms. The ones with `main()` entry points are registered as console scripts in `setup.py`.

## Testing

```bash
colcon test --packages-select dual_arm_sync
colcon test-result --verbose
```

`test/` contains the standard ament linters (`test_copyright.py`, `test_flake8.py`, `test_pep257.py`) and `test_hierarchical_bspline.py`, a standalone script that exercises collision detection and B-spline code with plots. `dual_arm_sync/test_pipeline.py` is a separate pipeline test script registered as `ros2 run dual_arm_sync test_pipeline`.

## Troubleshooting

**`No module named 'dual_arm_sync'`**: build and source the workspace.

```bash
cd ~/dual_arm_ws && colcon build --symlink-install && source install/setup.bash
```

**`Failed to find package 'dsr_gazebo2'` (or other `dsr_*`)**: clone `doosan-robot2` into `src/` and rebuild.

**"MESH COLLISION UNAVAILABLE -- FALLING BACK TO CAPSULES"**: install the mesh dependencies and build the hull file.

```bash
pip install python-fcl trimesh pycollada
python3 dual_arm_sync/mesh_collision.py --build /path/to/ros2_ws
```

Capsule results are not comparable with mesh results. Use `DUAL_ARM_REQUIRE_MESH=1` to make this a hard error.

**Stage fails with "file not found" for a JSON**: you are in a different directory than the previous stage, or skipped a stage. All stages use the current working directory.

**"no IK solution for at least one arm"** (151/161/171/251): the goal is outside the reachable region (reach roughly 0.35-1.25 m from the base for random goals) or collides at the endpoint. Try different goals, or `--force` in `step_251` to write the config anyway.

**Executor says no `joint_states` received**: Gazebo is not running, or the arm names/topics do not match. The executors expect `/{arm}/gz/joint_states` or `/{arm}/joint_states`.

**Planning succeeds but nothing moves**: the executors fall back to planning-only when Gazebo is not detected (`Gazebo: not detected -- planning-only run`). Start the matching launch file first.

**Timing alone cannot separate some pairs**: expected for tightly crossing goals; stage 5 (deformation + Kuramoto re-sync) handles these. Raise `DUAL_ARM_TIME_BUDGET_S` or `DUAL_ARM_ESCAPE_ITER` if it stops early.

## Known issues

- `step_136.py` writes `s126_execution.json` (the six-arm file name) instead of `s136_execution.json`, so running it can overwrite the six-arm report.
- `step_135.py` prints "step 124 resolved all by timing" in its pass-through message (cosmetic copy-paste).
- `package.xml` still has placeholder maintainer / URL fields (`Your Name`, `yourusername`).
- `step_251`-`step_256` print "6-arm cell, 1.39 m grid" but are pinned to the `dual` cell (two arms).
- `step_11.py` imports `dual_arm_sync.dual_arm_sync.step_1`, which only resolves with a particular package layout; running from the installed package may fail.

- The Stage 5 `stage_C` / `topo_sequential` fallbacks and several parameter defaults differ from the paper; see [Differences between the paper and this code](#differences-between-the-paper-and-this-code).

- `step_253` prints replay hints for `step_200.py --replay-pair`, which does not exist in this repository, and `step_256` prints its banner as "STEP 206" (copy-paste leftovers).

## References

1. *"Kuramoto Synchronized Configuration Space Trajectory Generation for an Ensemble of Manipulators Sharing a Common Workspace"*, ICRA 2027 submission (under review).
2. A. M. Kabir, A. Kanyuck, R. K. Malhan, A. V. Shembekar, S. Thakar, B. C. Shah, S. K. Gupta, "Generation of synchronized configuration space trajectories of multi-robot systems," ICRA 2019, pp. 8683-8690.
3. A. M. Kabir et al., "Generation of synchronized configuration space trajectories with workspace path constraints for an ensemble of robots," Int. J. Robot. Res., 40(2-3), 651-678, 2021.
4. Doosan Robotics M1013 documentation and `doosan-robot2` ROS 2 packages.

## Author

Designed, implemented and maintained by **[@sriramreddysangu](https://github.com/sriramreddysangu)**, IIT Tirupati (me24s501@iittp.ac.in).

All code in this package (kinematics and collision model, B-spline seeding, Kuramoto synchronizer, deformation stage, execution and benchmark harnesses) is the author's own work.
