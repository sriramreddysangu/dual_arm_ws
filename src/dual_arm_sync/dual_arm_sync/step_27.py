#!/usr/bin/env python3

import contextlib, io, json, os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robot6x as R
import step_26 as X26                     
from _bench_common import quiet, scan_all, summarise

TRIALS    = int(os.environ.get('DUAL_ARM_TRIALS', '30'))
SEED      = int(os.environ.get('DUAL_ARM_SEED', '0'))
DURATION  = float(os.environ.get('DUAL_ARM_DURATION', '10.0'))
CLEAR_M   = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
EE_TOL_MM = float(os.environ.get('DUAL_ARM_EE_TOL_MM', '50.0'))
EXEC_ON   = os.environ.get('DUAL_ARM_EXEC', '1') != '0'
SPEED     = max(0.5, float(os.environ.get('DUAL_ARM_EXEC_SPEED', '1.0')))
SETTLE_S  = float(os.environ.get('DUAL_ARM_SETTLE_S', '6.0'))
RATE_HZ   = 100.0
RAMP_S    = 2.0
SAME_CELL = [('dsr01', 'dsr02'), ('dsr03', 'dsr04'), ('dsr05', 'dsr06')]
HOME = np.zeros(6)
RECOVER = os.environ.get('DUAL_ARM_RECOVER', '1') != '0'


def start_collisions(start, names, bases):
    bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if R.pair_collides(start[a], bases[a], start[b], bases[b],
                               margin=CLEAR_M):
                bad.append(f'{a}<->{b}')
    return bad

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

ZONE = ((-1.45, 1.45), (-0.75, 0.75), (0.45, 0.95))
REACH_MIN, REACH_MAX = 0.35, 1.25          

def sample_target(rng, base):
    (x0, x1), (y0, y1), (z0, z1) = ZONE
    sh = np.asarray(base) + np.array([0.0, 0.0, 0.15])
    for _ in range(400):
        t = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1),
                      rng.uniform(z0, z1)])
        r = float(np.linalg.norm(t - sh))
        if REACH_MIN < r < REACH_MAX:
            return t
    return np.array([base[0], base[1] * 0.6, 0.7])


def side_swaps(tgt, bases):
    out = []
    for (a, b) in SAME_CELL:
        if float(tgt[a][1]) * float(bases[a][1]) < 0 and \
           float(tgt[b][1]) * float(bases[b][1]) < 0:
            out.append(f'{a}<->{b}')
    return out

if _ROS_OK:
    class HarnessNode(Node):
        def __init__(self, arm_names):
            super().__init__('step_27_harness')
            self._names = arm_names
            self._pubs = {n: self.create_publisher(
                Float64MultiArray, f'/{n}/gz/dsr_position_controller/commands', 10)
                for n in arm_names}
            self._cur = {n: None for n in arm_names}
            for name in arm_names:
                for t in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(JointState, t,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < R.NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, R.NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys) else np.array(msg.position[:R.NDOF]))
            self._cur[name] = q.astype(float)

        def wait_states(self, timeout=20.0):
            t0 = time.time()
            while time.time() - t0 < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._cur[n] is not None for n in self._names):
                    return True
            return False

        def joints(self):
            t0 = time.time()
            while time.time() - t0 < 0.3:
                rclpy.spin_once(self, timeout_sec=0.02)
            return {n: self._cur[n].copy() for n in self._names}

        def _send(self, poses):
            msg = Float64MultiArray()
            for n in self._names:
                msg.data = [float(v) for v in X26._cmd(np.asarray(poses[n]))]
                self._pubs[n].publish(msg)
            rclpy.spin_once(self, timeout_sec=0.)

        def ramp_to(self, goal, secs=RAMP_S):
            src = {n: (self._cur[n] if self._cur[n] is not None
                       else np.asarray(goal[n])).copy() for n in self._names}
            n_steps = max(2, int(secs * RATE_HZ)); dt = 1.0 / RATE_HZ
            for k in range(n_steps):
                s = k / (n_steps - 1); w = s * s * (3 - 2 * s)
                self._send({n: src[n] + w * (np.asarray(goal[n]) - src[n])
                            for n in self._names})
                time.sleep(dt)

        def recover_home(self, names, bases, secs=3.0):
            for n in names:
                goal = {m: (HOME.copy() if m == n else
                            (self._cur[m].copy() if self._cur[m] is not None
                             else HOME.copy())) for m in self._names}
                self.ramp_to(goal, secs=secs)
            return self.joints()

        def execute(self, arm_pos, duration, speed):
            n = max(len(arm_pos[m]) for m in self._names)
            wall = duration / speed
            n_out = max(2, int(wall * RATE_HZ)); dt = 1.0 / RATE_HZ
            idxs = np.linspace(0, n - 1, n_out).astype(int)
            for k in idxs:
                t0 = time.monotonic_ns()
                self._send({m: arm_pos[m][min(k, len(arm_pos[m]) - 1)]
                            for m in self._names})
                rem = int(1e9 * dt) - (time.monotonic_ns() - t0)
                if rem > 0: time.sleep(rem * 1e-9)
            cmd_final = {m: np.asarray(arm_pos[m][-1], float) for m in self._names}
            VEL_OK = 0.01
            POS_OK = float(os.environ.get('DUAL_ARM_SETTLE_DEG', '0.3'))
            POS_OK = np.radians(POS_OK)
            last = {m: None for m in self._names}; settled = 0
            t0 = time.time()
            while time.time() - t0 < SETTLE_S:
                self._send(cmd_final)
                mv = 0.0; perr = 0.0; ok = True
                for m in self._names:
                    if self._cur[m] is None:
                        ok = False; continue
                    if last[m] is not None:
                        mv = max(mv, float(np.max(np.abs(self._cur[m] - last[m])) / 0.01))
                    dq = np.angle(np.exp(1j * (self._cur[m] - cmd_final[m])))
                    perr = max(perr, float(np.max(np.abs(dq))))
                    last[m] = self._cur[m].copy()
                if ok and mv < VEL_OK and perr < POS_OK:
                    settled += 1
                    if settled >= 20: break
                else:
                    settled = 0
                time.sleep(0.01)
            return self.joints()

def one_trial(t, rng, mods, node):
    step_21, step_22, step_23, step_24, step_25 = mods
    names = list(R.ARM_NAMES)
    bases = {n: np.array(R.ROBOT_BASES[n]) for n in names}
    if node is not None:
        start = node.joints()
        bad = start_collisions(start, names, bases)
        if bad and RECOVER:
            print(f'         start pose COLLIDING {bad} -> parking at HOME')
            start = node.recover_home(names, bases)
            bad = start_collisions(start, names, bases)
        if bad:
            row = {'trial': t, 'category': 'START_COLLIDING', 'duration_s': None,
                   'overhead': None, 'clearance_cm': None, 'pairs': ';'.join(bad),
                   'reason': f'start pose already colliding {bad}',
                   'method': '', 'failed_stage': 'start_pose_colliding',
                   'side_swaps': '', 'executed': False, 'max_ee_err_mm': None,
                   'exec_success': False, 't21_s': None, 't2225_s': None,
                   'targets': {}, 'ee_err_mm': {}, 'start_joints': {},
                   'target_joints': {}, 'final_joints': {}}
            return row
    else:
        start = {n: np.zeros(R.NDOF) for n in names}
    tgt = {n: sample_target(rng, bases[n]) for n in names}
    row = {'trial': t, 'category': 'IK_FAIL', 'duration_s': None, 'overhead': None,
           'clearance_cm': None, 'pairs': '', 'reason': '', 'method': '',
           'failed_stage': '', 'side_swaps': ';'.join(side_swaps(tgt, bases)),
           'executed': False, 'max_ee_err_mm': None, 'exec_success': False,
           't21_s': None, 't2225_s': None,
           'targets': {n: np.round(tgt[n], 3).tolist() for n in names},
           'ee_err_mm': {}, 'final_joints': {},
           'start_joints': {n: np.round(start[n], 5).tolist() for n in names},
           'target_joints': {}}

    arms = {n: {'base': bases[n], 'start_q': start[n],
                'target_local': tgt[n] - bases[n], 'target_world': tgt[n].tolist(),
                'target_rot': None, 'constrain_orient': False, 'forced_q': None}
            for n in names}
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            best = step_21.greedy(arms)
    except Exception as e:
        row['reason'] = f'IK exception: {type(e).__name__}: {e}'
        row['failed_stage'] = 'step_21_exception'; return row
    row['t21_s'] = round(time.time() - t0, 1)
    if best is None or any(best.get(n) is None for n in names):
        row['reason'] = 'no IK solution'; row['failed_stage'] = 'step_21_ik'; return row
    endpoint_ok = bool(best.pop('_endpoint_ok', True))
    if not endpoint_ok:
        row['reason'] = 'no mesh-clear endpoint assignment (targets too close)'
        row['failed_stage'] = 'step_21_endpoint_infeasible'; return row

    row['target_joints'] = {n: np.round(np.asarray(best[n], float), 5).tolist()
                            for n in names}
    ik = {'duration': DURATION, 'arm_names': names, 'endpoint_ok': True}
    for n in names:
        ik[n] = {'base': bases[n].tolist(),
                 'start_joints': np.asarray(start[n]).tolist(),
                 'target_joints': np.asarray(best[n]).tolist(),
                 'end_joints': np.asarray(best[n]).tolist()}
    with open('s21_ik.json', 'w') as fh: json.dump(ik, fh)

    t1 = time.time()
    for mod, tag in ((step_22, 's22'), (step_23, 's23'),
                     (step_24, 's24'), (step_25, 's25')):
        good, _ = quiet(mod.main)
        if not good:
            row['category'] = 'UNRESOLVED'; row['reason'] = f'{tag} aborted'
            row['failed_stage'] = f'{tag}_abort'; return row
    row['t2225_s'] = round(time.time() - t1, 1)

    scan = json.load(open('s23_collision_map.json'))
    s24 = json.load(open('s24_resolved.json'))
    s25 = json.load(open('s25_synchronized.json'))
    row['pairs'] = ';'.join(p['pair'] for p in scan['pairs'] if p['status'] == 'COLLISION')
    row['method'] = (s25.get('synchronisation_report', {}) or {}).get('method', '')
    if scan['overall_status'] == 'SAFE':
        cat = 'NO_COLLISION'
    elif not s24.get('needs_retraction', True):
        cat = 'KURAMOTO'         
    elif s25['synchronisation_report']['collision_free']:
        cat = 'RETRACTION'       
    else:
        cat = 'UNRESOLVED'
        row['failed_stage'] = 'step_25_' + (row['method'] or 'budget')
        row['reason'] = 'residual collisions after budget'
    row['category'] = cat
    if cat == 'UNRESOLVED':
        return row

    dur = float(s25.get('final_duration_s') or s25.get('duration') or DURATION)
    P = {n: np.array(s25[n]['trajectory']['positions'], float) for n in names}
    bad, worst = scan_all(R, names, bases, P, CLEAR_M)
    row['duration_s'] = round(dur, 3)
    row['overhead'] = round(dur / DURATION - 1.0, 3)
    row['clearance_cm'] = round(worst * 100, 1)
    if bad:
        row['category'] = 'UNRESOLVED'; row['reason'] = f'post-check collision {bad}'
        row['failed_stage'] = 'post_verification'; return row

    if node is None or not EXEC_ON:
        row['reason'] = 'plan-only (no execution)'
        return row
    arm_pos = {}
    nout = max(2, int(round(dur * RATE_HZ)))
    for n in names:
        sin = np.linspace(0, 1, len(P[n])); sout = np.linspace(0, 1, nout)
        arm_pos[n] = np.clip(
            np.vstack([np.interp(sout, sin, P[n][:, j]) for j in range(R.NDOF)]).T,
            R.POS_LIM[:, 0], R.POS_LIM[:, 1])
    issues = X26.preflight(names, arm_pos, bases)
    if issues:
        row['category'] = 'UNRESOLVED'; row['failed_stage'] = 'preflight'
        row['reason'] = f'preflight {issues}'; return row
    node.ramp_to({n: arm_pos[n][0] for n in names})       
    final_q = node.execute(arm_pos, dur, SPEED)
    row['executed'] = True
    errs = {}
    for n in names:
        fq = final_q.get(n)
        if fq is None:
            errs[n] = float('inf'); continue
        ee = R.fk_world(np.asarray(fq, float), bases[n])
        errs[n] = float(np.linalg.norm(ee - tgt[n]) * 1000.0)
    row['ee_err_mm'] = {n: round(v, 1) for n, v in errs.items()}
    row['final_joints'] = {n: (np.round(final_q[n], 5).tolist()
                               if final_q.get(n) is not None else None)
                           for n in names}
    worst_arm = max(errs, key=errs.get)
    row['max_ee_err_mm'] = round(errs[worst_arm], 1)
    row['exec_success'] = errs[worst_arm] <= EE_TOL_MM
    if not row['exec_success']:
        row['reason'] = (f'EE error {errs[worst_arm]:.0f}mm on {worst_arm} '
                         f'(> {EE_TOL_MM:.0f}mm)')
        row['failed_stage'] = 'execution_ee_error'
    return row


def loophole_report(rows):
    print('\n' + '-' * 74)
    print('  LOOPHOLE REPORT')
    print('-' * 74)
    plan_ok = [r for r in rows
               if r['category'] in ('NO_COLLISION', 'KURAMOTO', 'RETRACTION')]
    execd = [r for r in rows if r['executed']]
    exec_ok = [r for r in execd if r['exec_success']]
    fails = [r for r in rows if r not in exec_ok]
    hs = {}
    for r in rows:
        if r['failed_stage']:
            hs[r['failed_stage']] = hs.get(r['failed_stage'], 0) + 1
    print('  failures by stage      : ' + (', '.join(
        f'{k}={v}' for k, v in sorted(hs.items(), key=lambda x: -x[1])) or 'none'))
    ms = {}
    for r in plan_ok:
        ms.setdefault(f"{r['category']}", []).append(r)
    for m, rs in sorted(ms.items(), key=lambda x: -len(x[1])):
        d = [x['duration_s'] for x in rs if x['duration_s']]
        print(f'  {m:14s}: n={len(rs):2d}' +
              (f'  mean_dur={np.mean(d):6.2f}s' if d else ''))
    if execd:
        errs = [r['max_ee_err_mm'] for r in execd if r['max_ee_err_mm'] is not None]
        print(f'  execution              : {len(exec_ok)}/{len(execd)} within '
              f'{EE_TOL_MM:.0f}mm   mean_err={np.mean(errs):.1f}mm  '
              f'worst={max(errs):.1f}mm')
    n_swap_fail = sum(1 for r in rows if r['side_swaps'] and r in fails)
    print(f'  side-swap trials       : '
          f'{sum(1 for r in rows if r["side_swaps"])}/{len(rows)}  '
          f'(among failures: {n_swap_fail})')
    print(f'  PLAN success           : {len(plan_ok)}/{len(rows)}  '
          f'({100.0 * len(plan_ok) / max(len(rows), 1):.0f}%)')
    print(f'  END-TO-END success     : {len(exec_ok)}/{len(rows)}  '
          f'({100.0 * len(exec_ok) / max(len(rows), 1):.0f}%)   '
          f'[plan clear AND EE <= {EE_TOL_MM:.0f}mm]')


def main():
    print('\n' + '=' * 74)
    print(f'  STEP 27  --  CLOSED-LOOP benchmark (plan + Gazebo execution), '
          f'{TRIALS} trials')
    print('=' * 74)
    import step_21, step_22, step_23, step_24, step_25
    mods = (step_21, step_22, step_23, step_24, step_25)
    node = None
    if EXEC_ON and _ROS_OK:
        rclpy.init()
        node = HarnessNode(list(R.ARM_NAMES))
        if node.wait_states(20.0):
            print('  Gazebo: LIVE (joint states received) -- trials plan from the')
            print('  present robot pose and EXECUTE with visualisation.')
        else:
            print('  Gazebo joint states NOT received -> plan-only mode.')
            try: node.destroy_node()
            except Exception: pass
            rclpy.shutdown(); node = None
    else:
        print('  Plan-only mode (rclpy unavailable or DUAL_ARM_EXEC=0).')
    rng = np.random.default_rng(SEED)
    rows = []; t0 = time.time(); cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='s27_')
    try:
        os.chdir(tmp)
        for t in range(1, TRIALS + 1):
            r = one_trial(t, rng, mods, node)
            rows.append(r)
            ok = 'OK ' if r['exec_success'] else ('-  ' if not r['executed'] else 'ERR')
            print(f'  [{t:3d}/{TRIALS}] {r["category"]:12s} {ok} '
                  f'ee={r["max_ee_err_mm"] if r["max_ee_err_mm"] is not None else "-":>7} '
                  f'dur={r["duration_s"] if r["duration_s"] else "-":>6} '
                  f'clr={r["clearance_cm"] if r["clearance_cm"] is not None else "-":>5} '
                  f'swap={"Y" if r["side_swaps"] else "-"} '
                  f'{(r["failed_stage"] or r["reason"])[:30]}', flush=True)
            if node is not None and RECOVER and not r['exec_success']:
                node.recover_home(list(R.ARM_NAMES),
                                  {n: np.array(R.ROBOT_BASES[n]) for n in R.ARM_NAMES})
            for f in ('s21_ik.json', 's22_trajectories.json', 's23_collision_map.json',
                      's24_resolved.json', 's25_synchronized.json'):
                if os.path.exists(f): os.remove(f)
    finally:
        os.chdir(cwd)
        if node is not None:
            try: node.destroy_node()
            except Exception: pass
            rclpy.shutdown()
    print(f'\n  wall time {time.time() - t0:.1f}s')
    cfg_path = os.path.join(cwd, 's27_config_results.json')
    try:
        with open(cfg_path, 'w') as fh:
            json.dump({'n_trials': len(rows), 'ee_tol_mm': EE_TOL_MM,
                       'trials': [{k: r.get(k) for k in
                                   ('trial', 'category', 'executed', 'exec_success',
                                    'max_ee_err_mm', 'ee_err_mm', 'duration_s',
                                    'clearance_cm', 'pairs', 'failed_stage',
                                    'targets', 'start_joints', 'target_joints',
                                    'final_joints')} for r in rows]}, fh, indent=2)
        print(f'  Saved: {cfg_path}')
    except Exception as e:
        print(f'  config dump failed: {e}')
    loophole_report(rows)
    summarise(rows, '6-ARM CLOSED-LOOP Kuramoto-first (plan+execute)',
              os.path.join(cwd, 's27'))


if __name__ == '__main__':
    main()
    