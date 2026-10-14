#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'four'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, time
import numpy as np
from scipy.optimize import minimize
from _robot8x import (NDOF, POS_LIM, VEL_LIM, fk, fk_world, pair_collides,
                      CLEAR_M, L2, L3, A)

W_POS, W_ROT = 1.0, 0.15
IK_KEEP = 40
IK_TOL_POS, IK_TOL_ROT, IK_UNIQ = 0.010, 0.050, 0.12
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
    r2 = r_xy - A; dist = np.sqrt(r2**2 + pz**2)
    c3 = np.clip((dist**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
    for th3 in [np.arccos(c3), -np.arccos(c3)]:
        for wrist in [0, _PI_2, -_PI_2]:
            for base in [t1, t1 + _PI_2, t1 - _PI_2, t1 + np.pi, t1 - np.pi]:
                th2 = np.arctan2(pz, r2) - np.arctan2(L3 * np.sin(th3),
                                                      L2 + L3 * np.cos(th3))
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
        if use_ori: e += W_ROT * rot_err(T[:3, :3], target_rot)**2
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
        if all(float(np.linalg.norm(q - v)) >= IK_UNIQ for v in valid):
            valid.append(q)
    valid.sort(key=lambda q: (round(float(np.max(np.abs(nearest_wrap(q, current) - current) / VEL_LIM)), 3),
                              float(np.sum(JOINT_COST_W * np.abs(nearest_wrap(q, current) - current)))))
    return valid[:IK_KEEP]


def repair_endpoint_collisions(best, arms_data, restarts=8, iters=400, seed=0):
    names = list(best.keys())
    bases = {n: arms_data[n]["base"] for n in names}
    starts = {n: arms_data[n]["start_q"] for n in names}
    cand = {n: [nearest_wrap(q, starts[n]) for q in arms_data[n]["_sols"]] for n in names}
    travel = {n: np.array([float(np.sum(JOINT_COST_W * np.abs(q - starts[n]))) for q in cand[n]])
              for n in names}
    optime = {n: np.array([float(np.max(np.abs(q - starts[n]) / VEL_LIM))
                           for q in cand[n]]) for n in names}

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
                c += int(tab[(name, o)][ki, asg[o]]) if (name, o) in tab \
                     else int(tab[(o, name)][asg[o], ki])
            return c
        def total(asg):
            return sum(int(tab[(names[i], nj)][asg[names[i]], asg[nj]])
                       for i in range(len(names)) for nj in names[i + 1:])
        best_asg = None; best_key = None
        for r in range(restarts):
            asg = {n: (int(min(range(len(cand[n])),
                               key=lambda k: (round(optime[n][k], 3), travel[n][k])))
                       if r == 0 else int(rng.integers(len(cand[n])))) for n in names}
            for _ in range(iters):
                conf = [n for n in names if coll_of(n, asg[n], asg) > 0]
                if not conf: break
                n = conf[int(rng.integers(len(conf)))]
                asg[n] = min(range(len(cand[n])),
                             key=lambda k: (coll_of(n, k, asg),
                                            round(optime[n][k], 3), travel[n][k]))
            for _ in range(20):
                moved = False
                for n in names:
                    ks = [k for k in range(len(cand[n])) if coll_of(n, k, asg) == 0]
                    if ks:
                        k2 = min(ks, key=lambda k: (round(optime[n][k], 3), travel[n][k]))
                        if k2 != asg[n]: asg[n] = k2; moved = True
                if not moved: break
            tc = total(asg)
            sys_optime = max(float(optime[n][asg[n]]) for n in names)
            tt = float(sum(travel[n][asg[n]] for n in names))
            if best_key is None or (tc, round(sys_optime, 3), tt) < best_key:
                best_key = (tc, round(sys_optime, 3), tt); best_asg = dict(asg)
            if best_key[0] == 0: break
        final_asg, final_key = best_asg, best_key
        if best_key[0] == 0: break
    for n in names:
        best[n] = cand[n][final_asg[n]]
    endpoint_ok = final_key[0] == 0
    if not endpoint_ok:
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                if pair_collides(best[ni], bases[ni], best[nj], bases[nj],
                                 margin=CLEAR_M):
                    print(f"  [warn] targets {ni}<->{nj} closer than the gate "
                          f"(no collision-free IK branch combination exists)")
    best['_endpoint_ok'] = endpoint_ok
    return best


def lin_path(s, e, n=5):
    return np.array([s + f * (e - s) for f in np.linspace(0., 1., n)])


def score(q, start):
    qw = nearest_wrap(q, start)
    travel = float(np.sum(JOINT_COST_W * np.abs(qw - start)))
    t_op = float(np.max(np.abs(qw - start) / VEL_LIM))
    return (round(t_op, 3), round(travel, 3)), qw


def greedy(arms_data):
    best = {}
    for name in arms_data:
        d = arms_data[name]
        if d.get('forced_q') is not None:
            q = np.array(d['forced_q'], float); best[name] = q
            d['_sols'] = [q]; d['n_sols'] = 1; continue
        sols = solve_ik(d['target_local'], d['target_rot'], d['start_q'],
                        constrain_orient=d.get('constrain_orient', False))
        if not sols: return None
        d['_sols'] = sols
        scored = sorted((score(q, d['start_q']) for q in sols), key=lambda x: x[0])
        best[name] = scored[0][1]; d['n_sols'] = len(sols)
    return repair_endpoint_collisions(best, arms_data)

def is_manual_mode():
    if os.environ.get('DUAL_ARM_S61_MODE') is not None:
        return True
    return any(k.startswith('DUAL_ARM_') and k.endswith('_GOAL') for k in os.environ)


def manual_arms_data(names, bases, cur_q):
    mode = os.environ.get('DUAL_ARM_S61_MODE', 'cartesian').strip().lower()
    if mode not in ('cartesian', 'joint'):
        raise ValueError(f"DUAL_ARM_S61_MODE must be 'joint' or 'cartesian', got {mode!r}")
    arms = {}
    for n in names:
        key = n.upper()
        start_env = os.environ.get(f'DUAL_ARM_{key}_START')
        if start_env:
            vals = [float(x) for x in start_env.split(',')]
            if len(vals) != NDOF:
                raise ValueError(f'DUAL_ARM_{key}_START needs {NDOF} degree values, got {len(vals)}')
            start_q = np.radians(np.array(vals, float))
        else:
            start_q = cur_q[n].copy()
        goal_env = os.environ.get(f'DUAL_ARM_{key}_GOAL')
        if goal_env is None:
            raise ValueError(f'DUAL_ARM_{key}_GOAL is required in manual mode '
                             f'(arm {n} has no override set)')
        vals = [float(x) for x in goal_env.split(',')]
        if mode == 'joint':
            if len(vals) != NDOF:
                raise ValueError(f'DUAL_ARM_{key}_GOAL needs {NDOF} joint values '
                                 f'in degrees for joint mode, got {len(vals)}')
            forced_q = np.radians(np.array(vals, float))
            arms[n] = {'base': bases[n], 'start_q': start_q, 'target_local': None,
                       'target_rot': None, 'constrain_orient': False,
                       'forced_q': forced_q}
        else:
            if len(vals) != 3:
                raise ValueError(f'DUAL_ARM_{key}_GOAL needs x,y,z world metres '
                                 f'for cartesian mode, got {len(vals)} values')
            tgt_world = np.array(vals, float)
            arms[n] = {'base': bases[n], 'start_q': start_q,
                       'target_local': tgt_world - np.asarray(bases[n]),
                       'target_rot': None, 'constrain_orient': False,
                       'forced_q': None}
    return arms

ZONE_XY = (-1.25, 1.25)
ZONE_Y = (-0.75, 0.75)
ZONE_Z = (0.35, 1.15)
REACH_MIN, REACH_MAX = 0.35, 1.25


def sample_goal(rng, base):
    shoulder = np.asarray(base) + np.array([0, 0, 0.15])
    for _ in range(400):
        t = np.array([rng.uniform(*ZONE_XY), rng.uniform(*ZONE_Y),
                      rng.uniform(*ZONE_Z)])
        r = float(np.linalg.norm(t - shoulder))
        if REACH_MIN < r < REACH_MAX: return t
    return np.array([base[0], base[1] * 0.6, 0.7])


def auto_arms_data(names, bases, cur_q, seed):
    rng = np.random.default_rng(seed)
    arms = {}
    for n in names:
        tgt = sample_goal(rng, bases[n])
        arms[n] = {'base': bases[n], 'start_q': cur_q[n].copy(),
                   'target_local': tgt - np.asarray(bases[n]),
                   'target_rot': None, 'constrain_orient': False,
                   'forced_q': None}
    return arms


def main():
    names = list(_R8X.ARM_NAMES); bases = {n: np.array(_R8X.ROBOT_BASES[n]) for n in names}
    cur_q = {n: np.zeros(NDOF) for n in names}   # HOME unless *_START overrides it
    manual = is_manual_mode()
    if manual:
        print(f"  STEP 161 (manual trial): mode="
             f"{os.environ.get('DUAL_ARM_S61_MODE', 'cartesian')}")
        arms = manual_arms_data(names, bases, cur_q)
    else:
        seed = int(os.environ.get('DUAL_ARM_SEED', '0'))
        print(f"  STEP 161 (auto trial): random goals, seed={seed}")
        print(f"    (set DUAL_ARM_S61_MODE=cartesian|joint and "
              f"DUAL_ARM_<ARM>_GOAL=... to script an exact scenario instead)")
        arms = auto_arms_data(names, bases, cur_q, seed)
    t0 = time.time()
    best = greedy(arms)
    if best is None:
        print("  STEP 161: no IK solution for at least one arm -- aborting.")
        sys.exit(1)
    endpoint_ok = best.pop('_endpoint_ok', True)
    if not endpoint_ok:
        print("  STEP 161: [warn] no mesh-clear endpoint combination exists for "
              "these goals; writing the least-colliding assignment "
              "anyway so you can see why (step_163 will flag it).")

    ik = {'duration': 10.0, 'arm_names': names, 'endpoint_ok': bool(endpoint_ok)}
    cart = {}
    for n in names:
        s = arms[n]['start_q']; e = best[n]
        ik[n] = {'base': list(bases[n]), 'start_joints': s.tolist(),
                 'target_joints': e.tolist(), 'end_joints': e.tolist()}
        cart[n] = {'start_world': fk_world(s, bases[n]).tolist(),
                   'target_world': fk_world(e, bases[n]).tolist()}
    ik['cartesian'] = cart
    with open('s161_ik.json', 'w') as fh:
        json.dump(ik, fh)
    print(f'  Saved: s161_ik.json  ({time.time()-t0:.2f}s)')
    for n in names:
        sw = np.round(cart[n]['start_world'], 4).tolist()
        tw = np.round(cart[n]['target_world'], 4).tolist()
        print(f'    [{n}] start(world)={sw}  ->  target(world)={tw}')
    print('  Next: step_162 (or run step_167 for the full trial sweep).')


if __name__ == '__main__':
    main()