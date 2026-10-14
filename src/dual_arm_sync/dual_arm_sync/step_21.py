#!/usr/bin/env python3

import json, os, sys, time
from typing import Dict, Optional
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ROBOT_BASES, ARM_NAMES,
                      fk, fk_world, pair_min_dist, pair_collides)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

W_POS, W_ROT = 1.0, 0.15
IK_KEEP = 40
IK_TOL_POS, IK_TOL_ROT, IK_UNIQ = 0.010, 0.050, 0.12
N_PATH_SMP = 5
JOINT_COST_W = np.array([1.0, 1.5, 1.5, 0.6, 0.6, 0.4])
_PI_2 = np.pi / 2.0


def rot_err(Rg, Rw):
    return float(np.arccos(np.clip((np.trace(Rg.T @ Rw) - 1.0) / 2.0, -1.0, 1.0)))


def _ik_seeds(current, target_local):
    seeds = [current.copy()]
    for j in range(NDOF):
        for d in [0.3, -0.3, 0.6, -0.6]:
            s = current.copy(); s[j] += d
            seeds.append(np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1]))
    px, py, pz = target_local
    t1 = np.arctan2(py, px); r_xy = np.sqrt(px**2 + py**2)
    from _robot6x import L2, L3, A
    r2 = r_xy - A; dist = np.sqrt(r2**2 + pz**2)
    c3 = np.clip((dist**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
    for th3 in [np.arccos(c3), -np.arccos(c3)]:
        for wrist in [0, _PI_2, -_PI_2]:
            for base in [t1, t1 + _PI_2, t1 - _PI_2, t1 + np.pi, t1 - np.pi]:
                th2 = np.arctan2(pz, r2) - np.arctan2(L3 * np.sin(th3), L2 + L3 * np.cos(th3))
                seeds.append(np.clip(np.array([base, th2, th3, 0., wrist, 0.]),
                                     POS_LIM[:, 0], POS_LIM[:, 1]))
    return seeds


def nearest_wrap(q, start):
    qw = q.copy()
    for j in range(NDOF):
        cands = [q[j] + 2*np.pi*k for k in (-2, -1, 0, 1, 2)]
        cands = [c for c in cands if POS_LIM[j, 0] <= c <= POS_LIM[j, 1]]
        if cands: qw[j] = min(cands, key=lambda c: abs(c - start[j]))
    return qw


def solve_ik(target_local, target_rot, current, constrain_orient=False):
    use_ori = bool(constrain_orient and target_rot is not None)
    valid = []
    def obj(q):
        p, T = fk(q)
        e = np.sum((p - target_local)**2)
        if use_ori:
            e += W_ROT * rot_err(T[:3, :3], target_rot)**2
        return e
    for seed in _ik_seeds(current, target_local):
        res = minimize(obj, seed, method='SLSQP',
                       bounds=list(zip(POS_LIM[:, 0], POS_LIM[:, 1])),
                       options={'maxiter': 400, 'ftol': 1e-9})
        if not res.success: continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        p, T = fk(q)
        if float(np.linalg.norm(p - target_local)) > IK_TOL_POS: continue
        if use_ori and float(rot_err(T[:3, :3], target_rot)) > IK_TOL_ROT: continue
        if all(float(np.linalg.norm(q - v)) >= IK_UNIQ for v in valid): valid.append(q)
    valid.sort(key=lambda q: (round(float(np.max(np.abs(nearest_wrap(q, current) - current) / VEL_LIM)), 3),
                              float(np.sum(JOINT_COST_W * np.abs(nearest_wrap(q, current) - current)))))
    return valid[:IK_KEEP]


def lin_path(s, e, n=N_PATH_SMP):
    return np.array([s + f * (e - s) for f in np.linspace(0., 1., n)])


def op_time(qw, start):
    return float(np.max(np.abs(qw - start) / VEL_LIM))


def score(q, start, base, prior=None):
    qw = nearest_wrap(q, start)
    travel = float(np.sum(JOINT_COST_W * np.abs(qw - start)))
    t_op = op_time(qw, start)
    return (round(t_op, 3), round(travel, 3)), qw, {
        'op_time_s': round(t_op, 3), 'weighted_travel_rad': round(travel, 4),
        'max_joint_excursion_deg': round(float(np.max(np.abs(np.degrees(qw - start)))), 1)}


def repair_endpoint_collisions(best, arms_data, restarts=8, iters=400, seed=0):
    names = list(best.keys())
    bases = {n: arms_data[n]["base"] for n in names}
    starts = {n: arms_data[n]["start_q"] for n in names}
    cand = {n: [nearest_wrap(q, starts[n]) for q in arms_data[n]["_sols"]] for n in names}
    travel = {n: np.array([float(np.sum(JOINT_COST_W * np.abs(q - starts[n]))) for q in cand[n]])
              for n in names}
    optime = {n: np.array([float(np.max(np.abs(q - starts[n]) / VEL_LIM)) for q in cand[n]])
              for n in names}
    CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))

    def build_tab(margin):
        t = {}
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                M = np.zeros((len(cand[ni]), len(cand[nj])), bool)
                for a in range(len(cand[ni])):
                    for b in range(len(cand[nj])):
                        M[a, b] = pair_collides(cand[ni][a], bases[ni],
                                                cand[nj][b], bases[nj],
                                                margin=margin)
                t[(ni, nj)] = M
        return t

    rng = np.random.default_rng(seed)
    final_asg = None; final_key = None
    for margin in (CLEAR_M + 0.02, CLEAR_M):
        tab = build_tab(margin)
        def coll_of(name, ki, asg):
            c = 0
            for o in names:
                if o == name: continue
                c += int(tab[(name, o)][ki, asg[o]]) if (name, o) in tab else int(tab[(o, name)][asg[o], ki])
            return c
        def total(asg):
            return sum(int(tab[(names[i], nj)][asg[names[i]], asg[nj]])
                       for i in range(len(names)) for nj in names[i + 1:])
        best_asg = None; best_key = None
        for r in range(restarts):
            asg = {n: (int(min(range(len(cand[n])), key=lambda k: (round(optime[n][k], 3), travel[n][k]))) if r == 0 else int(rng.integers(len(cand[n])))) for n in names}
            for _ in range(iters):
                conf = [n for n in names if coll_of(n, asg[n], asg) > 0]
                if not conf: break
                n = conf[int(rng.integers(len(conf)))]
                asg[n] = min(range(len(cand[n])), key=lambda k: (coll_of(n, k, asg), round(optime[n][k], 3), travel[n][k]))
            for _ in range(20):
                moved = False
                for n in names:
                    ks = [k for k in range(len(cand[n])) if coll_of(n, k, asg) == 0]
                    if ks:
                        k2 = min(ks, key=lambda k: (round(optime[n][k], 3), travel[n][k]))
                        if k2 != asg[n]:
                            asg[n] = k2; moved = True
                if not moved: break
            tc = total(asg)
            sys_optime = max(float(optime[n][asg[n]]) for n in names)
            tt = float(sum(travel[n][asg[n]] for n in names))
            if best_key is None or (tc, round(sys_optime, 3), tt) < best_key:
                best_key = (tc, round(sys_optime, 3), tt); best_asg = dict(asg)
            if best_key[0] == 0: break
        final_asg, final_key = best_asg, best_key
        if best_key[0] == 0:
            break                        
    for n in names:
        best[n] = cand[n][final_asg[n]]
    endpoint_ok = final_key[0] == 0
    if endpoint_ok:
        print("  [ok] all target poses mutually mesh-clear")
    else:
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                if pair_collides(best[ni], bases[ni], best[nj], bases[nj]):
                    print(f"  [warn] targets {ni}<->{nj} closer than the gate "
                          f"(no collision-free IK branch combination exists)")
    best['_endpoint_ok'] = endpoint_ok
    return best


def greedy(arms_data):
    best = {}; prior = []
    for name in arms_data:
        d = arms_data[name]
        if d.get('forced_q') is not None:             
            q = np.array(d['forced_q'], float); best[name] = q
            d['_sols'] = [q]; d['n_sols'] = 1
            d['pos_err_mm'] = (round(float(np.linalg.norm(fk(q)[0] - d['target_local'])) * 1000, 3)
                               if d.get('target_local') is not None else 0.0)
            d['joint_aware_score'] = {'op_time_s': round(op_time(nearest_wrap(q, d['start_q']), d['start_q']), 3),
                                      'weighted_travel_rad': 0.0,
                                      'max_joint_excursion_deg': round(float(np.max(np.abs(np.degrees(q - d['start_q'])))), 1)}
            print(f'  [{name}] FORCED joint config (IK bypassed)')
            prior.append((lin_path(d['start_q'], q), d['base'])); continue
        con = d.get('constrain_orient', False)
        sols = solve_ik(d['target_local'], d['target_rot'], d['start_q'], constrain_orient=con)
        if not sols:
            if con:
                print(f'  [{name}] FAIL: no IK at requested orientation '
                      f'(use Enter=FREE orientation, or a reachable quaternion)')
            else:
                print(f'  [{name}] FAIL: no IK')
            return None
        d['_sols'] = sols
        scored = sorted((score(q, d['start_q'], d['base'], prior) for q in sols),
                        key=lambda x: x[0])
        sc, q, brk = scored[0]
        best[name] = q; d['joint_aware_score'] = brk; d['n_sols'] = len(sols)
        d['pos_err_mm'] = round(float(np.linalg.norm(fk(q)[0] - d['target_local'])) * 1000, 3)
        print(f'  [{name}] {len(sols)} sol(s)  op_time={brk["op_time_s"]:.2f}s  '
              f'travel={brk["weighted_travel_rad"]:.3f}rad  '
              f'max_exc={brk["max_joint_excursion_deg"]:.0f}deg  '
              f'pos_err={d["pos_err_mm"]:.2f}mm')
        prior.append((lin_path(d['start_q'], q), d['base']))
    best = repair_endpoint_collisions(best, arms_data)
    print('  --- final selection (after endpoint collision resolution) ---')
    for name in arms_data:
        d = arms_data[name]; s = d['start_q']; qe = nearest_wrap(best[name], s)
        otime = float(np.max(np.abs(qe - s) / VEL_LIM))
        mexc  = float(np.max(np.abs(np.degrees(qe - s))))
        trav  = float(np.sum(JOINT_COST_W * np.abs(qe - s)))
        print(f'  [{name}] FINAL: op_time={otime:.2f}s  max_exc={mexc:.0f}deg'
              f'  travel={trav:.2f}rad')
        d['final_op_time_s'] = round(otime, 3); d['final_max_exc_deg'] = round(mexc, 1)
    return best


if _ROS_OK:
    class JointReader(Node):
        def __init__(self):
            super().__init__('step_21_reader')
            self._q = {n: np.zeros(NDOF) for n in ARM_NAMES}
            self._ready = {n: False for n in ARM_NAMES}
            for name in ARM_NAMES:
                for topic in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)
        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys) else np.array(msg.position[:NDOF]))
            self._q[name] = q.astype(float); self._ready[name] = True
        def joints(self, n): return self._q[n].copy()
        def wait(self, t=20.):
            import time as _t; t0 = _t.time()
            while rclpy.ok() and _t.time()-t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._ready.values()): return True
            return False


def main(args=None):
    print('\n' + '='*66)
    print('  STEP 21  --  Joint-Aware IK Selection [6 ARM]')
    print('='*66)
    cur_q = {n: np.zeros(NDOF) for n in ARM_NAMES}; node = None
    if _ROS_OK:
        rclpy.init(args=args); node = JointReader()
        print('\n  Waiting for Gazebo joint states...')
        if node.wait(20.):
            for n in ARM_NAMES: cur_q[n] = node.joints(n)
            print('  All arms ready.')
        else:
            print('\n  !! GAZEBO JOINT STATES NOT RECEIVED (20s). Planning from the')
            print('  !! ZERO pose -- results may not match the running robot.')
            print('  !! Verify: ros2 topic echo /dsr01/gz/joint_states --once\n')
    arms_data = {}
    for name in ARM_NAMES:
        base = ROBOT_BASES[name]
        print(f'\n  ARM: {name.upper()}  base={base.tolist()}')
        while True:
            raw = input(f'  [{name}] TARGET pos X Y Z (m): ').strip()
            try: xyz = [float(v) for v in raw.replace(',', ' ').split()]
            except ValueError: continue
            if len(xyz) == 3: break
        tw = np.array(xyz)
        raw_q = input(f'  [{name}] TARGET quaternion w x y z (Enter=FREE orientation): ').strip()
        if raw_q == '':
            R = None; constrain = False
        else:
            constrain = True
            try:
                w, x, y, z = [float(v) for v in raw_q.replace(',', ' ').split()]
                n2 = w*w + x*x + y*y + z*z
                if n2 < 1e-9:
                    R = np.eye(3)
                else:
                    w, x, y, z = [v/np.sqrt(n2) for v in [w, x, y, z]]
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            except ValueError:
                R = np.eye(3)
        raw_j = input(f'  [{name}] TARGET joints j1..j6 rad (Enter=solve IK): ').strip()
        forced_q = None
        if raw_j != '':
            try:
                vals = [float(v) for v in raw_j.replace(',', ' ').split()]
                if len(vals) == NDOF: forced_q = np.array(vals)
            except ValueError: forced_q = None
        arms_data[name] = {'base': base, 'start_q': cur_q[name],
            'target_world': tw.tolist(), 'target_local': tw - base, 'target_rot': R,
            'constrain_orient': constrain, 'forced_q': forced_q}
    dur_raw = input('\n  Duration [s] (Enter=10.0): ').strip()
    duration = float(dur_raw) if dur_raw else 10.0
    print('\n  Solving IK (greedy joint-aware)...')
    t0 = time.time(); best = greedy(arms_data); ik_ms = round((time.time()-t0)*1000, 1)
    if best is None: print('\n  FAIL'); sys.exit(1)
    endpoint_ok = bool(best.pop('_endpoint_ok', True))
    out = {'duration': duration, 'arm_names': ARM_NAMES, 'ik_total_time_ms': ik_ms,
           'endpoint_ok': endpoint_ok,
           'selection_method': 'greedy_pairwise_joint_aware_mesh_csp'}
    for name in ARM_NAMES:
        d = arms_data[name]; q = best[name]; ee = fk_world(q, d['base'])
        out[name] = {'base': np.array(d['base']).tolist(), 'start_joints': d['start_q'].tolist(),
            'target_joints': q.tolist(), 'target_joints_deg': np.degrees(q).tolist(),
            'end_joints': q.tolist(), 'target_world': d['target_world'],
            'target_ee_world': ee.tolist(), 'position_error_mm': d['pos_err_mm'],
            'n_solutions': d['n_sols'], 'joint_aware_score': d['joint_aware_score']}
    with open('s21_ik.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  IK total: {ik_ms:.0f} ms   Saved: s21_ik.json')
    print(f'  Next : ros2 run dual_arm_sync step_22\n')
    if node:
        try: node.destroy_node()
        except: pass
        rclpy.shutdown()


if __name__ == '__main__': main()