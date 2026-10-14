#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'nine'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import contextlib, io, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import step_131 as X1
import step_132 as X2
import step_133 as X3
import step_134 as X4
import step_135 as X5
import step_136 as X6
from _robot8x import (ARM_NAMES, ROBOT_BASES, NDOF, RATE_HZ, CLEAR_M,
                      pair_min_dist, fk_world, resample)

N_TRIALS = int(os.environ.get('DUAL_ARM_TRIALS', '20'))
SEED = int(os.environ.get('DUAL_ARM_SEED', '0'))
DO_EXEC = os.environ.get('DUAL_ARM_EXEC', '1') != '0'
EE_TOL_MM = float(os.environ.get('DUAL_ARM_EE_TOL_MM', '50.0'))
RECOVER = os.environ.get('DUAL_ARM_RECOVER', '1') != '0'
HOME = np.zeros(NDOF)
ZONE_XY = (-1.25, 1.25)         
ZONE_Y = (-0.75, 0.75)
ZONE_Z = (0.35, 1.15)
REACH_MIN, REACH_MAX = 0.35, 1.25


def quiet(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        return True, ''
    except SystemExit:
        return False, 'abort'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def sample_goal(rng, base):
    shoulder = np.asarray(base) + np.array([0, 0, 0.15])
    for _ in range(400):
        t = np.array([rng.uniform(*ZONE_XY), rng.uniform(*ZONE_Y),
                      rng.uniform(*ZONE_Z)])
        r = float(np.linalg.norm(t - shoulder))
        if REACH_MIN < r < REACH_MAX: return t
    return np.array([base[0], base[1] * 0.6, 0.7])


def start_collisions(joints, names, bases):
    bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if pair_min_dist(joints[a], bases[a], joints[b], bases[b]) < CLEAR_M:
                bad.append(f'{a}<->{b}')
    return bad


def build_arms_data(rng, cur_q, names, bases):
    arms = {}
    for n in names:
        tgt = sample_goal(rng, bases[n])
        arms[n] = {'base': bases[n], 'start_q': cur_q[n].copy(),
                   'target_local': tgt - np.asarray(bases[n]),
                   'target_rot': None, 'constrain_orient': False,
                   'forced_q': None}
    return arms


class HarnessNode:
    def __init__(self, names, bases):
        self._names = names; self._bases = bases
        self._cur = {n: HOME.copy() for n in names}
        self._live = False
        self._rclpy = None; self._node = None
        if not DO_EXEC:
            return
        try:
            import rclpy
            rclpy.init()
            self._rclpy = rclpy
            self._node = X6.ExecNode(names)     
            t0 = time.time()
            while time.time() - t0 < 5.0:
                rclpy.spin_once(self._node, timeout_sec=0.1)
            missing = [n for n in names if self._node._cur_q[n] is None]
            if missing:
                print(f'         [gazebo] no joint state received for {missing} '
                      f'within 5s -- check topic names/QoS on those arms')
                self._live = False
            else:
                self._live = True
                for n in names: self._cur[n] = self._node._cur_q[n].copy()
        except Exception as e:
            print(f'         [gazebo] connection failed: '
                  f'{type(e).__name__}: {e}')
            self._live = False

    @property
    def live(self):
        return self._live

    def joints(self):
        if self._live:
            for n in self._names:
                if self._node._cur_q[n] is not None:
                    self._cur[n] = self._node._cur_q[n].copy()
        return {n: self._cur[n].copy() for n in self._names}

    def ramp_to(self, goal, secs=3.0):
        if not self._live:
            for n in self._names: self._cur[n] = goal[n].copy()
            return
        from std_msgs.msg import Float64MultiArray
        dt = 1.0 / RATE_HZ; n_steps = max(2, int(secs * RATE_HZ))
        src = self.joints()
        msg = Float64MultiArray()
        for k in range(n_steps):
            f = k / (n_steps - 1); w = f * f * (3 - 2 * f)
            for n in self._names:
                q = src[n] + w * (goal[n] - src[n])
                msg.data = [float(v) for v in q]
                self._node._pubs[n].publish(msg)
            self._rclpy.spin_once(self._node, timeout_sec=0.)
            time.sleep(dt)
        for n in self._names: self._cur[n] = goal[n].copy()

    def recover_home(self, names, bases, secs=3.0):
        live = self.joints()
        fold = {m: np.array([live[m][0], 0., 0., 0., 0., 0.]) for m in self._names}
        self.ramp_to(fold, secs=secs)
        time.sleep(0.3)
        self.ramp_to({m: HOME.copy() for m in self._names}, secs=secs)
        st = self.joints()
        if not start_collisions(st, list(self._names), bases):
            return st
        print('         simultaneous recovery did not clear -> sequential')
        for order in (list(names), list(reversed(names))):
            for nme in order:
                cur = self.joints()
                held = {m: cur[m].copy() for m in self._names if m != nme}
                self.ramp_to({**held, nme: HOME.copy()}, secs=secs)
            st = self.joints()
            if not start_collisions(st, list(self._names), bases):
                return st
        return self.joints()

    def execute(self, arm_pos, duration):
        return self._node.execute(arm_pos, duration)

    def shutdown(self):
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass

_TIMING = {}


def one_trial(t, rng, names, bases, node):
    _TIMING.clear()
    t_wall0 = time.time()
    row = _one_trial_inner(t, rng, names, bases, node)
    row['t_ik_s'] = _TIMING.get('ik')
    row['t_seed_s'] = _TIMING.get('seed')
    row['t_scan_s'] = _TIMING.get('scan')
    row['t_kuramoto_s'] = _TIMING.get('kuramoto')
    row['t_escape_s'] = _TIMING.get('escape')
    row['t_planning_total_s'] = round(time.time() - t_wall0, 4)
    return row


def _one_trial_inner(t, rng, names, bases, node):
    cur_q = node.joints()
    bad = start_collisions(cur_q, names, bases)
    if bad:
        row = {'trial': t, 'category': 'START_COLLIDING', 'duration_s': None,
              'overhead': None, 'clearance_cm': None, 'pairs': ';'.join(bad),
              'reason': f'start pose already colliding {bad}',
              'method': '', 'failed_stage': 'start_pose_colliding',
              'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
              'start_joints': {n: np.round(cur_q[n], 5).tolist() for n in names},
              'target_joints': {}, 'final_joints': {}, 'settle': None}
        return row

    arms = build_arms_data(rng, cur_q, names, bases)
    t0 = time.time()
    best = X1.greedy(arms)
    t1 = time.time()
    _TIMING['ik'] = round(t1 - t0, 4)
    if best is None:
        return {'trial': t, 'category': 'IK_FAIL', 'duration_s': None,
                'overhead': None, 'clearance_cm': None, 'pairs': '',
                'reason': 'no IK solution for at least one arm',
                'method': '', 'failed_stage': 'step_131_ik_fail',
                'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
                'start_joints': {n: cur_q[n].tolist() for n in names},
                'target_joints': {}, 'final_joints': {}, 'settle': None}
    endpoint_ok = best.pop('_endpoint_ok', True)
    if not endpoint_ok:
        return {'trial': t, 'category': 'IK_FAIL', 'duration_s': None,
                'overhead': None, 'clearance_cm': None, 'pairs': '',
                'reason': 'no mesh-clear endpoint assignment (targets too close)',
                'method': '', 'failed_stage': 'step_131_endpoint_infeasible',
                'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
                'start_joints': {n: cur_q[n].tolist() for n in names},
                'target_joints': {n: best[n].tolist() for n in names},
                'final_joints': {}, 'settle': None}

    ik = {'duration': 10.0, 'arm_names': names, 'endpoint_ok': True}
    for n in names:
        ik[n] = {'base': list(bases[n]), 'start_joints': cur_q[n].tolist(),
                 'target_joints': best[n].tolist(), 'end_joints': best[n].tolist()}
    json.dump(ik, open('s131_ik.json', 'w'))

    _stage_role = ['seed', 'scan', 'kuramoto', 'escape']
    for _si, (mod, tag) in enumerate(((X2, 's122'), (X3, 's123'), (X4, 's124'), (X5, 's125'))):
        _ts = time.time()
        ok, err = quiet(mod.main)
        _TIMING[_stage_role[_si]] = round(time.time() - _ts, 4)
        if not ok:
            return {'trial': t, 'category': 'STAGE_ERROR', 'duration_s': None,
                    'overhead': None, 'clearance_cm': None, 'pairs': '',
                    'reason': err, 'method': '', 'failed_stage': f'{tag}_error',
                    'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
                    'start_joints': {n: cur_q[n].tolist() for n in names},
                    'target_joints': {n: best[n].tolist() for n in names},
                    'final_joints': {}, 'settle': None}

    s5 = json.load(open('s135_synchronized.json'))
    resolved = s5.get('final_verification', {}).get('verified_collision_free', False)
    method = s5.get('synchronisation_report', {}).get('method', '')
    dur = float(s5.get('duration', 10.0))
    ov = float(s5.get('duration_overhead', 0.0))
    if not resolved:
        return {'trial': t, 'category': 'UNRESOLVED', 'duration_s': round(dur, 3),
                'overhead': round(ov, 3), 'clearance_cm': None, 'pairs': '',
                'reason': 'residual collisions after budget', 'method': method,
                'failed_stage': 'step_135_unresolved',
                'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
                'start_joints': {n: cur_q[n].tolist() for n in names},
                'target_joints': {n: best[n].tolist() for n in names},
                'final_joints': {}, 'settle': None}

    category = ('NO_COLLISION' if method == 'none_needed' else
                'KURAMOTO' if method == 'kuramoto_phase_lag' else
                'RETRACTION')
    arm_pos = {n: resample(np.array(s5[n]['trajectory']['positions'], float), dur)
              for n in names}
    issues = X6.preflight(names, arm_pos, bases)
    if issues:
        return {'trial': t, 'category': 'UNRESOLVED', 'duration_s': round(dur, 3),
                'overhead': round(ov, 3), 'clearance_cm': None,
                'pairs': ';'.join(issues), 'reason': f'preflight {issues}',
                'method': method, 'failed_stage': 'preflight',
                'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
                'start_joints': {n: cur_q[n].tolist() for n in names},
                'target_joints': {n: best[n].tolist() for n in names},
                'final_joints': {}, 'settle': None}

    row = {'trial': t, 'category': category, 'duration_s': round(dur, 3),
          'overhead': round(ov, 3), 'clearance_cm': None, 'pairs': '',
          'reason': '', 'method': method, 'failed_stage': '',
          'start_joints': {n: cur_q[n].tolist() for n in names},
          'target_joints': {n: best[n].tolist() for n in names}}

    if not node.live:
        row['executed'] = False; row['exec_success'] = None
        row['final_joints'] = {}; row['max_ee_err_mm'] = None; row['settle'] = None
        node_final = {n: best[n].copy() for n in names}  
        for n in names: node._cur[n] = node_final[n].copy()
        return row

    node.ramp_to({n: arm_pos[n][0] for n in names})      
    exec_res = node.execute(arm_pos, dur)
    row['executed'] = True
    row['settle'] = exec_res.get('settle')
    raw_final = exec_res.get('final_joints', {}) or {}
    live_now = node.joints()          
    final_q = {}
    for n in names:
        v = raw_final.get(n)
        final_q[n] = np.asarray(v, float) if v is not None else live_now[n]
    worst = 0.0
    for n in names:
        tgt_ee = fk_world(best[n], np.asarray(bases[n]))
        got_ee = fk_world(final_q[n], np.asarray(bases[n]))
        worst = max(worst, float(np.linalg.norm(got_ee - tgt_ee) * 1000))
    row['max_ee_err_mm'] = round(worst, 3)
    row['exec_success'] = bool(exec_res.get('success')) and worst <= EE_TOL_MM
    row['final_joints'] = {n: final_q[n].tolist() for n in names}
    return row


def summarise(rows, title, prefix):
    import csv
    cnt = {}
    for r in rows: cnt[r['category']] = cnt.get(r['category'], 0) + 1
    n_plan_ok = sum(1 for r in rows
                   if r['category'] in ('NO_COLLISION', 'KURAMOTO', 'RETRACTION'))
    n_e2e_ok = sum(1 for r in rows if r.get('exec_success'))
    print(f'\n  {title}: {cnt}')
    print(f'  PLAN success    : {n_plan_ok}/{len(rows)} '
          f'({100*n_plan_ok/max(len(rows),1):.0f}%)')
    print(f'  END-TO-END OK   : {n_e2e_ok}/{len(rows)} '
          f'({100*n_e2e_ok/max(len(rows),1):.0f}%)')
    with open(prefix + '_results.json', 'w') as fh:
        json.dump({'n_trials': len(rows), 'counts': cnt, 'trials': rows}, fh,
                  indent=2, default=str)
    keys = ['trial', 'category', 'duration_s', 'overhead', 'clearance_cm',
           'pairs', 'reason', 'method', 'failed_stage', 'executed',
           'exec_success', 'max_ee_err_mm',
           't_ik_s', 't_seed_s', 't_scan_s', 't_kuramoto_s', 't_escape_s',
           't_planning_total_s']
    with open(prefix + '_results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow(r)


def main():
    print('=' * 74)
    print(f'  STEP 137  --  CLOSED-LOOP benchmark, {N_TRIALS} trials, cell="9-arm"')
    print('=' * 74)
    names = list(ARM_NAMES); bases = {n: np.array(ROBOT_BASES[n]) for n in names}
    rng = np.random.default_rng(SEED)
    node = HarnessNode(names, bases)
    print('  Gazebo:', 'LIVE' if node.live else 'not detected -- planning-only run')
    rows = []; t0 = time.time(); consec_sc = 0
    for t in range(1, N_TRIALS + 1):
        r = one_trial(t, rng, names, bases, node)
        rows.append(r)
        tag = 'OK ' if r.get('exec_success') else ('-  ' if r.get('exec_success') is None else 'FAIL')
        print(f'  [{t:3d}/{N_TRIALS}] {r["category"]:14s} {tag} '
              f'dur={r.get("duration_s","-")} err={r.get("max_ee_err_mm","-")}')
        if r['category'] == 'START_COLLIDING':
            consec_sc += 1
            if node.live and consec_sc == 3:
                print('         [breaker] 3 consecutive START_COLLIDING -> recovery pass')
                node.recover_home(names, bases)
            if consec_sc >= 6:
                print('         [breaker] wedge persists -- stopping run early'); break
        else:
            consec_sc = 0
        if RECOVER and node.live and not r.get('exec_success', True):
            _bad = start_collisions(node.joints(), names, bases)
            if _bad:
                print(f'         recovering (start colliding {_bad})')
                node.recover_home(names, bases)
        for f in ('s131_ik.json', 's132_trajectories.json',
                  's133_collision_map.json', 's134_resolved.json',
                  's135_synchronized.json'):
            try: os.remove(f)
            except OSError: pass
    print(f'\n  wall time {time.time()-t0:.1f}s')
    summarise(rows, f"9-arm CELL closed-loop ({len(rows)} trials)",
             's137_config')
    node.shutdown()


if __name__ == '__main__':
    main()