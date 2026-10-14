#!/usr/bin/env python3

import json
import os
import sys

import numpy as np
from scipy.interpolate import splrep, splev

try:
    from dual_arm_sync.ik_solver import (
        forward_kinematics,  
        RobotBases,           
        JOINT_LIMITS,        
        JOINT_VEL_MAX,       
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        forward_kinematics, RobotBases, JOINT_LIMITS, JOINT_VEL_MAX,
    )

JOINT_ACC_MAX = JOINT_VEL_MAX * 2.0  

N_EVAL = 1000   

_TAU_TABLE = np.linspace(0.0, 1.0, 200_000)
_S_TABLE   = (10 * _TAU_TABLE**3
              - 15 * _TAU_TABLE**4
              + 6  * _TAU_TABLE**5)


def _minjerk_tau(s: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(s, 0.0, 1.0), _S_TABLE, _TAU_TABLE)


def s_to_time(s_values: np.ndarray, T: float) -> np.ndarray:
    tau = _minjerk_tau(np.asarray(s_values, dtype=float))
    t   = tau * T
    eps = 1e-8
    for i in range(1, len(t)):
        if t[i] <= t[i - 1] + eps:
            t[i] = t[i - 1] + eps
    return t

def fit_bspline(joint_wps: np.ndarray,
                t_wps:     np.ndarray,
                T:         float,
                n_eval:    int = N_EVAL) -> dict:
    n_way = len(joint_wps)
    if n_way < 2:
        raise ValueError(f"Need ≥ 2 waypoints, got {n_way}")

    t_n    = t_wps / T                       
    t_ev_n = np.linspace(0.0, 1.0, n_eval)
    t_ev   = t_ev_n * T

    pos = np.zeros((n_eval, 6))
    vel = np.zeros((n_eval, 6))
    acc = np.zeros((n_eval, 6))

    for j in range(6):
        k   = min(3, n_way - 1)            
        tck = splrep(t_n, joint_wps[:, j], k=k, s=0)
        pos[:, j] = splev(t_ev_n, tck, der=0)
        vel[:, j] = splev(t_ev_n, tck, der=1) / T        
        acc[:, j] = splev(t_ev_n, tck, der=2) / (T ** 2)

    return {
        'time':            t_ev,
        'positions':       pos,
        'velocities':      vel,
        'accelerations':   acc,
        'joint_waypoints': joint_wps.copy(),
        'waypoint_times':  t_wps.copy(),
        'num_samples':     n_eval,
    }

def scale_duration_if_needed(traj:      dict,
                               joint_wps: np.ndarray,
                               t_wps:     np.ndarray,
                               T:         float) -> dict:
    v_scale = 1.0
    a_scale = 1.0
    for j in range(6):
        v_peak = float(np.max(np.abs(traj['velocities'][:, j])))
        a_peak = float(np.max(np.abs(traj['accelerations'][:, j])))
        if v_peak > JOINT_VEL_MAX[j]:
            v_scale = max(v_scale, v_peak / JOINT_VEL_MAX[j])
        if a_peak > JOINT_ACC_MAX[j]:
            a_scale = max(a_scale, np.sqrt(a_peak / JOINT_ACC_MAX[j]))

    scale = max(v_scale, a_scale)
    if scale > 1.0:
        T_new = T * scale * 1.05          # 5 % headroom
        t_new = t_wps * (T_new / T)
        print(f"    ⚠  Limit violation — scaling  {T:.3f}s → {T_new:.3f}s"
              f"  (vel×{v_scale:.2f}  acc×{a_scale:.2f})")
        return fit_bspline(joint_wps, t_new, T_new)

    print(f"    ✓  All limits satisfied  (T = {T:.3f}s unchanged)")
    return traj

def build_arm_trajectory(robot_name:  str,
                          joint_wps:  np.ndarray,
                          s_values:   np.ndarray,
                          q_start:    np.ndarray,
                          q_end:      np.ndarray,
                          p_start:    np.ndarray,
                          p_end:      np.ndarray,
                          base:       np.ndarray,
                          T:          float) -> dict:
    N = len(joint_wps)
    print(f"\n  ── {robot_name.upper()} ──")
    dist_mm = float(np.linalg.norm(p_end - p_start)) * 1000
    dq_deg  = float(np.max(np.degrees(np.abs(q_end - q_start))))
    print(f"  Cartesian path    : {np.round(p_start, 4)}  →  {np.round(p_end, 4)}")
    print(f"  Path length       : {dist_mm:.1f} mm")
    print(f"  Chained IK pts    : {N}  (from Stage 1 — NOT re-solved here)")
    print(f"  Max joint Δ       : {dq_deg:.1f}°")

    wps = joint_wps.copy()
    wps[0]  = q_start
    wps[-1] = q_end

    t_wps = s_to_time(np.asarray(s_values, dtype=float), T)
    print(f"  Time range        : [{t_wps[0]:.4f}, {t_wps[-1]:.4f}] s"
          f"  (min-jerk, non-uniform)")

    traj = fit_bspline(wps, t_wps, T)

    traj = scale_duration_if_needed(traj, wps, t_wps, T)

    final_T = float(traj['time'][-1])

    p_local, *_ = forward_kinematics(traj['positions'][-1])
    err_mm = float(np.linalg.norm(p_local + base - p_end)) * 1000
    print(f"  EE end error      : {err_mm:.2f} mm")

    v_util = np.max(np.abs(traj['velocities']), axis=0) / JOINT_VEL_MAX * 100
    print(f"  Vel utilisation % : {np.round(v_util, 1)}")
    print(f"  Final duration    : {final_T:.3f}s")

    traj['robot_name'] = robot_name
    traj['metadata'] = {
        'robot_name':        robot_name,
        'duration':          final_T,
        'num_samples':       N_EVAL,
        'num_waypoints':     N,
        'start_joints':      q_start.tolist(),
        'end_joints':        q_end.tolist(),
        'start_world_pos':   p_start.tolist(),
        'target_world_pos':  p_end.tolist(),
        'planning_method':   'cartesian_minjerk_bspline_independent',
        'ee_error_mm':       round(err_mm, 3),
    }
    return traj

def make_arm_output(traj: dict) -> dict:
    t   = traj['time']         
    pos = traj['positions']    
    vel = traj['velocities']
    acc = traj['accelerations']

    traj_pts = [
        {'time': float(t[i]), 'joints': pos[i].tolist()}
        for i in range(len(t))
    ]

    return {
        'trajectory_points': traj_pts,

        'time':            t.tolist(),
        'positions':       pos.tolist(),
        'velocities':      vel.tolist(),
        'accelerations':   acc.tolist(),
        'num_samples':     traj['num_samples'],

        'joint_waypoints': traj['joint_waypoints'].tolist(),
        'waypoint_times':  traj['waypoint_times'].tolist(),

        'metadata': traj['metadata'],
        'duration': float(traj['time'][-1]),
    }

def _extract_path_samples(sol: dict, arm_id: str):
    ps = sol.get('ik_path_samples')
    if ps is not None and ps.get('n_success', 0) >= 2:
        js = np.array(ps['joint_solutions'], dtype=float)   # (N, 6)
        sv = np.array(ps['s_values'],        dtype=float)   # (N,)
        print(f"   {arm_id}: {ps['n_success']}/{ps['n_samples']} IK samples  "
              f"curvature_N={ps.get('curvature_n_adaptive', ps['n_samples'])}")
        return js, sv

    print(f"   {arm_id}: ⚠  no ik_path_samples — using 2-point fallback")
    q_s = np.array(sol['current_joints'], dtype=float)
    q_e = np.array(sol['optimal_joints'],  dtype=float)
    return np.vstack([q_s, q_e]), np.array([0.0, 1.0])


def _pad_to_duration(traj: dict, T_new: float) -> dict:
    T_old = float(traj['time'][-1])
    if T_old >= T_new - 1e-9:
        return traj
    wps   = traj['joint_waypoints']
    t_wps = traj['waypoint_times'] * (T_new / T_old)
    updated = fit_bspline(wps, t_wps, T_new)
    updated['robot_name'] = traj['robot_name']
    updated['metadata']   = {**traj['metadata'], 'duration': T_new}
    return updated

def main():
    print("\n" + "=" * 80)
    print("TRAJECTORY GENERATION  —  Cartesian Min-Jerk B-Spline  (Paper §IV-C)")
    print("=" * 80)
    print("  Paper method   : Cartesian min-jerk path + IK + B-spline  (§IV-C)")
    print("  Independence   : Each arm planned separately — no cross-arm NLP")
    print("  IK source      : ik_path_samples from dual_arm_ik_solver (Stage 1)")
    print("  Time param     : s(τ) = 10τ³ − 15τ⁴ + 6τ⁵  inverted → t_i")
    print("  Synchronisation: deferred to kuramoto_synchronization (Stage 4)")
    print("  Input          : ik_solutions.json")
    print("  Output         : trajectories.json")
    print("=" * 80)

    if not os.path.exists('ik_solutions.json'):
        print("\n❌  ik_solutions.json not found")
        print("    Run: ros2 run dual_arm_sync dual_arm_ik_solver")
        return

    with open('ik_solutions.json') as f:
        ik = json.load(f)

    sol1 = ik.get('dsr01')
    sol2 = ik.get('dsr02')
    if sol1 is None or sol2 is None:
        print("\n❌  ik_solutions.json missing 'dsr01' or 'dsr02'")
        return

    T_req  = float(ik.get('duration', 10.0))
    method = ik.get('method', 'unknown')
    print(f"\n✓  ik_solutions.json loaded")
    print(f"   IK method    : {method}")
    print(f"   Requested T  : {T_req:.1f}s")

    base1 = RobotBases.DSR01_BASE   
    base2 = RobotBases.DSR02_BASE    

    js1, sv1 = _extract_path_samples(sol1, 'dsr01')
    js2, sv2 = _extract_path_samples(sol2, 'dsr02')

    print("\n" + "─" * 80)
    print("Building trajectories  (each arm independent — paper §IV-C without central NLP)")
    print("─" * 80)

    traj1 = build_arm_trajectory(
        robot_name = 'dsr01',
        joint_wps  = js1,
        s_values   = sv1,
        q_start    = np.array(sol1['current_joints'],   dtype=float),
        q_end      = np.array(sol1['optimal_joints'],   dtype=float),
        p_start    = np.array(sol1['current_world_pos'], dtype=float),
        p_end      = np.array(sol1['target_world_pos'],  dtype=float),
        base       = base1,
        T          = T_req,
    )

    traj2 = build_arm_trajectory(
        robot_name = 'dsr02',
        joint_wps  = js2,
        s_values   = sv2,
        q_start    = np.array(sol2['current_joints'],   dtype=float),
        q_end      = np.array(sol2['optimal_joints'],   dtype=float),
        p_start    = np.array(sol2['current_world_pos'], dtype=float),
        p_end      = np.array(sol2['target_world_pos'],  dtype=float),
        base       = base2,
        T          = T_req,
    )

    T1    = float(traj1['time'][-1])
    T2    = float(traj2['time'][-1])
    T_max = max(T1, T2)
    print(f"\n  Duration alignment : dsr01={T1:.3f}s  dsr02={T2:.3f}s  → common={T_max:.3f}s")
    traj1 = _pad_to_duration(traj1, T_max)
    traj2 = _pad_to_duration(traj2, T_max)

    output = {
        'arm_ids':  ['dsr01', 'dsr02'],
        'duration': T_max,
        'method':   'cartesian_minjerk_bspline_independent',
        'dsr01':    make_arm_output(traj1),
        'dsr02':    make_arm_output(traj2),
    }

    with open('trajectories.json', 'w') as f:
        json.dump(output, f, indent=2)

    kb = os.path.getsize('trajectories.json') / 1024
    print(f"\n{'─' * 80}")
    print(f"✓  Saved: trajectories.json  ({kb:.0f} KB)")
    print(f"   dsr01 : {N_EVAL} samples  T={traj1['time'][-1]:.3f}s  "
          f"waypoints={len(traj1['joint_waypoints'])}")
    print(f"   dsr02 : {N_EVAL} samples  T={traj2['time'][-1]:.3f}s  "
          f"waypoints={len(traj2['joint_waypoints'])}")
    print(f"\n{'=' * 80}")
    print("Next step: ros2 run dual_arm_sync collision_checker")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()