#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'dual'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, os, sys, time
import numpy as np
from _robot8x import (NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, DH_TO_CMD,
                      fk_world, pair_collides, RATE_HZ, CLEAR_M, PREFILTER_M,
                      _caps_surface, resample)

HOLD_S = 1.5
SETTLE_S = float(os.environ.get('DUAL_ARM_SETTLE_S', '0'))   
CTRL_TAU = float(os.environ.get('DUAL_ARM_CTRL_TAU', '5.3'))
SETTLE_MIN = float(os.environ.get('DUAL_ARM_SETTLE_MIN', '8.0'))
SETTLE_MAX = float(os.environ.get('DUAL_ARM_SETTLE_MAX', '45.0'))
EE_TOL_MM = float(os.environ.get('DUAL_ARM_EE_TOL_MM', '25.0'))
JT_TOL_D = 5.0
REORDER_CMD = False        
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    from rclpy.qos import qos_profile_sensor_data
    _ROS_OK = True
except ImportError:
    _ROS_OK = False


def _cmd(q):
    return q[DH_TO_CMD] if REORDER_CMD else q


def settle_budget(arm_pos, duration, tol_rad):
    if SETTLE_S > 0:
        return SETTLE_S
    v_peak = 0.0
    for n, P in arm_pos.items():
        P = np.asarray(P, float)
        if len(P) < 2: continue
        dt = duration / max(len(P) - 1, 1)
        v_peak = max(v_peak, float(np.max(np.abs(np.diff(P, axis=0))) / dt))
    e0 = max(v_peak * CTRL_TAU, tol_rad)
    t_need = CTRL_TAU * float(np.log(max(e0 / max(tol_rad, 1e-6), 1.0))) + 2.0
    return float(np.clip(t_need, SETTLE_MIN, SETTLE_MAX))


def preflight(arm_names, arm_pos, bases):
    issues = []; N = len(arm_names)
    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = arm_names[i], arm_names[j]; pi, pj = arm_pos[ni], arm_pos[nj]
            nc = 0
            for k in range(min(len(pi), len(pj))):
                if _caps_surface(pi[k], bases[ni], pj[k], bases[nj]) > PREFILTER_M:
                    continue
                if pair_collides(pi[k], bases[ni], pj[k], bases[nj], margin=CLEAR_M):
                    nc += 1
            if nc > 0: issues.append(f'{ni}<->{nj}: {nc} collision steps')
    return issues


def verify_arm(name, ref_q, final_q):
    base = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
    ref_q = np.array(ref_q, float)
    tgt_ee = fk_world(ref_q, base)
    res = {'commanded_final_deg': np.degrees(ref_q).tolist()}
    if final_q is not None:
        fq = np.array(final_q, float); ee = fk_world(fq, base)
        dq = np.angle(np.exp(1j * (fq - ref_q)))
        res.update({'final_joints_deg': np.degrees(fq).tolist(),
                    'joint_error_deg': round(float(np.max(np.abs(np.degrees(dq)))), 4),
                    'ee_error_mm': round(float(np.linalg.norm(ee - tgt_ee) * 1000), 3)})
        res['ee_ok'] = res['ee_error_mm'] <= EE_TOL_MM
        res['joints_ok'] = res['joint_error_deg'] <= JT_TOL_D
    return res


if _ROS_OK:
    class ExecNode(Node):
        def __init__(self, arm_names):
            super().__init__('dual_arm_executor')
            self._names = arm_names
            self._pubs = {n: self.create_publisher(
                Float64MultiArray, f'/{n}/gz/dsr_position_controller/commands', 10)
                for n in arm_names}
            self._cur_q = {n: None for n in arm_names}
            for name in arm_names:
                for t in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(JointState, t,
                        lambda msg, n=name: self._cb(msg, n),
                        qos_profile_sensor_data)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys) else np.array(msg.position[:NDOF]))
            self._cur_q[name] = q.astype(float)

        def execute(self, arm_pos, duration):
            dt_ns = int(1e9 / RATE_HZ)
            n_steps = max(len(arm_pos[n]) for n in self._names)
            msg = Float64MultiArray(); t0_wall = time.monotonic()
            for k in range(n_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    idx = min(k, len(arm_pos[name]) - 1)
                    msg.data = [float(v) for v in _cmd(arm_pos[name][idx])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0: time.sleep(rem * 1e-9)
            for _ in range(int(HOLD_S * RATE_HZ)):
                t0 = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in _cmd(arm_pos[name][-1])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0: time.sleep(rem * 1e-9)
            wall_s = time.monotonic() - t0_wall
            VEL_SETTLED = 0.01
            POS_SETTLED = np.radians(float(os.environ.get('DUAL_ARM_SETTLE_DEG', '0.3')))
            cmd_final = {n: np.array(arm_pos[n][-1], float) for n in self._names}
            budget = settle_budget(arm_pos, duration, POS_SETTLED)
            t_settle = time.time()
            e_first = None; t_first = None
            final_q = {n: None for n in self._names}
            last_q = {n: None for n in self._names}
            settled = 0; exit_perr = None
            while time.time() - t_settle < budget:
                t0s = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in _cmd(arm_pos[name][-1])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                mv = 0.0; perr = 0.0; have_all = True
                for n in self._names:
                    if self._cur_q[n] is None: have_all = False; continue
                    if last_q[n] is not None:
                        mv = max(mv, float(np.max(np.abs(self._cur_q[n] - last_q[n])) / 0.01))
                    dq = np.angle(np.exp(1j * (self._cur_q[n] - cmd_final[n])))
                    perr = max(perr, float(np.max(np.abs(dq))))
                    last_q[n] = self._cur_q[n].copy()
                exit_perr = perr
                if have_all and e_first is None and time.time() - t_settle > 1.0:
                    e_first = perr; t_first = time.time()
                if have_all and mv < VEL_SETTLED and perr < POS_SETTLED:
                    settled += 1
                    if settled >= 20: break
                else: settled = 0
                rem = dt_ns - (time.monotonic_ns() - t0s)
                if rem > 0: time.sleep(rem * 1e-9)
            for n in self._names:
                if self._cur_q[n] is not None: final_q[n] = self._cur_q[n].copy()
            el = time.time() - t_settle
            tau_est = None
            if e_first and exit_perr and exit_perr > 0 and e_first > exit_perr:
                dt_meas = time.time() - t_first
                if dt_meas > 0.5:
                    tau_est = round(float(dt_meas / np.log(e_first / exit_perr)), 2)
            return {'success': True, 'mode': 'ros2', 'steps_sent': n_steps,
                    'wall_time_s': round(wall_s, 3),
                    'settle': {'settle_s': round(el, 2),
                               'budget_s': round(budget, 2),
                               'converged': bool(settled >= 20),
                               'hit_cap': bool(settled < 20 and el >= budget - 0.1),
                               'tau_est_s': tau_est,
                               'residual_deg': (round(float(np.degrees(exit_perr)), 3)
                                                if exit_perr is not None else None)},
                    'final_joints': {n: final_q[n].tolist() if final_q[n] is not None
                                     else None for n in self._names}, 'error': None}


def dry_run(arm_names, arm_pos, duration):
    n = max(len(arm_pos[nm]) for nm in arm_names)
    print(f'\n  DRY-RUN: {n} steps ({duration:.2f}s @ {RATE_HZ:.0f} Hz)')
    return {'success': True, 'mode': 'dry_run', 'steps_sent': n, 'wall_time_s': None,
            'settle': None,
            'final_joints': {nm: arm_pos[nm][-1].tolist() for nm in arm_names},
            'error': None}


def main(args=None):
    with open('s105_synchronized.json') as fh: sdata = json.load(fh)
    arm_names = sdata.get('arm_names', ARM_NAMES); duration = float(sdata['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {}
    for name in arm_names:
        pos = np.array(sdata[name]['trajectory']['positions'], float)
        arm_pos[name] = resample(pos, duration)
    print(f'\n  STEP 106: arms {arm_names} dur={duration:.2f}s')
    issues = preflight(arm_names, arm_pos, bases)
    if issues: print(f'  PRE-FLIGHT FAIL: {issues}'); sys.exit(1)
    print('  Pre-flight OK (real mesh)')
    if _ROS_OK:
        rclpy.init(args=args); node = ExecNode(arm_names)
        try: exec_res = node.execute(arm_pos, duration)
        finally:
            try: node.destroy_node()
            except Exception: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)
    verif = {n: verify_arm(n, arm_pos[n][-1], exec_res['final_joints'].get(n))
             for n in arm_names}
    report = {'arm_names': arm_names, 'duration_s': duration,
              'execution': exec_res, 'verification': verif,
              'pipeline': 'kuramoto-first + perpendicular mesh escape'}
    for name in arm_names:
        v = verif[name]
        print(f'  [{name}] j_err={v.get("joint_error_deg","?")} deg '
              f'ee_err={v.get("ee_error_mm","?")} mm ok={v.get("ee_ok","?")}')
    with open('s106_execution.json', 'w') as fh: json.dump(report, fh, indent=2)
    print('  Saved: s106_execution.json')


if __name__ == '__main__':
    main()