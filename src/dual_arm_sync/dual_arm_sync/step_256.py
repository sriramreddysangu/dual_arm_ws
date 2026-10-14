#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'dual'
os.environ.setdefault('DUAL_ARM_REQUIRE_MESH', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, time
import numpy as np
from _robot8x import (VEL_LIM, RATE_HZ, ROBOT_BASES, ARM_NAMES, resample,
                      fk_world)
import step_106 as S106

DO_APPROACH = os.environ.get('DUAL_ARM_S256_APPROACH', '1') != '0'
APPROACH_MIN_S = float(os.environ.get('DUAL_ARM_S256_APPROACH_MIN_S', '1.5'))
APPROACH_FRAC = 0.4          
GAP_WARN_DEG = 10.0

if S106._ROS_OK:
    import rclpy
    from std_msgs.msg import Float64MultiArray


CARTESIAN_SAMPLES = int(os.environ.get('DUAL_ARM_CARTESIAN_SAMPLES', '60'))


def cartesian_samples(arm_pos, bases, names, n_samples=None):
    if n_samples is None:
        n_samples = CARTESIAN_SAMPLES
    if n_samples <= 0:
        return {}
    out = {}
    for n in names:
        P = np.asarray(arm_pos[n], float)
        if len(P) == 0:
            out[n] = []
            continue
        idx = np.linspace(0, len(P) - 1, min(n_samples, len(P))).astype(int)
        out[n] = [np.round(fk_world(P[k], bases[n]), 5).tolist() for k in idx]
    return out

def approach_start(node, arm_pos, names):
    dt_ns = int(1e9 / RATE_HZ)
    for _ in range(int(0.5 * RATE_HZ)):
        rclpy.spin_once(node, timeout_sec=0.02)
    cur = {n: node._cur_q.get(n) for n in names}
    if any(cur[n] is None for n in names):
        missing = [n for n in names if cur[n] is None]
        print('  APPROACH: no live pose for {} -- skipping the ramp.'.format(
            ', '.join(missing)))
        return False

    gaps = {n: float(np.max(np.abs(np.degrees(arm_pos[n][0] - cur[n]))))
            for n in names}
    gap = max(gaps.values())
    if gap < 0.05:
        print('  APPROACH: arms already on the planned start (gap {:.3f} deg).'
              .format(gap))
        return True

    dur = APPROACH_MIN_S
    for n in names:
        disp = np.abs(arm_pos[n][0] - cur[n])
        dur = max(dur, float(np.max(disp / (VEL_LIM * APPROACH_FRAC))))
    print('  APPROACH: ramping to the planned start, max gap {:.1f} deg over '
          '{:.1f}s'.format(gap, dur))
    for n in names:
        if gaps[n] > GAP_WARN_DEG:
            print('    {} is {:.1f} deg from its planned start'.format(n, gaps[n]))

    msg = Float64MultiArray()
    n_app = max(2, int(dur * RATE_HZ))
    for k in range(n_app):
        t0 = time.monotonic_ns()
        a = k / (n_app - 1)
        for n in names:
            q = cur[n] + a * (arm_pos[n][0] - cur[n])
            msg.data = [float(v) for v in S106._cmd(q)]
            node._pubs[n].publish(msg)
        rclpy.spin_once(node, timeout_sec=0.)
        rem = dt_ns - (time.monotonic_ns() - t0)
        if rem > 0:
            time.sleep(rem * 1e-9)
    return True

def main(args=None):
    with open('s255_synchronized.json') as fh:
        sdata = json.load(fh)
    arm_names = sdata.get('arm_names', ARM_NAMES)
    duration = float(sdata['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {n: resample(np.array(sdata[n]['trajectory']['positions'], float),
                           duration) for n in arm_names}

    print('\n  STEP 206: arms {} dur={:.2f}s'.format(arm_names, duration))
    if not sdata.get('final_verification', {}).get('verified_collision_free', False):
        print('  REFUSING: step_255 did not verify this plan collision-free.')
        sys.exit(1)
    issues = S106.preflight(arm_names, arm_pos, bases)
    if issues:
        print('  PRE-FLIGHT FAIL: {}'.format(issues))
        sys.exit(1)
    print('  Pre-flight OK (real mesh)')

    if S106._ROS_OK:
        rclpy.init(args=args)
        node = S106.ExecNode(arm_names)
        try:
            if DO_APPROACH:
                approach_start(node, arm_pos, arm_names)
            exec_res = node.execute(arm_pos, duration)
        finally:
            try: node.destroy_node()
            except Exception: pass
            rclpy.shutdown()
    else:
        exec_res = S106.dry_run(arm_names, arm_pos, duration)

    verif = {n: S106.verify_arm(n, arm_pos[n][-1],
                                exec_res['final_joints'].get(n))
             for n in arm_names}
    report = {'arm_names': arm_names, 'duration_s': duration,
              'execution': exec_res, 'verification': verif,
              'pipeline': 'kuramoto-first + perpendicular mesh escape '
                          '(200-series single configuration)',
              'method': sdata.get('synchronisation_report', {}).get('method'),
              'duration_overhead': sdata.get('duration_overhead')}

    report['cartesian_resolved'] = cartesian_samples(arm_pos, bases, arm_names)
    naive = {}
    try:
        with open('s252_trajectories.json') as fh:
            raw = json.load(fh)
        raw_pos = {n: np.array(raw[n]['trajectory']['positions'], float)
                   for n in arm_names if n in raw}
        naive = cartesian_samples(raw_pos, bases, arm_names)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        naive = {}
    report['cartesian_naive'] = naive
    try:
        with open('s253_collision_map.json') as fh:
            report['naive_collision_map'] = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        with open('s251_config.json') as fh:
            report['configuration'] = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    for name in arm_names:
        v = verif[name]
        print('  [{}] j_err={} deg  ee_err={} mm  ok={}'.format(
            name, v.get('joint_error_deg', '?'), v.get('ee_error_mm', '?'),
            v.get('ee_ok', '?')))
    st = exec_res.get('settle') or {}
    if st:
        print('  settle {:.1f}s (budget {:.1f}s) converged={} residual={} deg'
              .format(st.get('settle_s', 0), st.get('budget_s', 0),
                      st.get('converged'), st.get('residual_deg')))

    with open('s256_execution.json', 'w') as fh:
        json.dump(report, fh, indent=2)
    print('  Saved: s256_execution.json'
          + ('  [{} arms cartesian, {} samples/arm]'.format(
              len(report['cartesian_resolved']), CARTESIAN_SAMPLES)
             if CARTESIAN_SAMPLES > 0 else ''))


if __name__ == '__main__':
    main()