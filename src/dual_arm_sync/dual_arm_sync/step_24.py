#!/usr/bin/env python3

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                      ARM_NAMES, pair_min_dist, pair_collides,
                      _arm_caps, caps_min_dist, LINK_RADII, SAFETY_MARGIN)

CLEAR_M  = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
DT       = 0.01
MIN_SAFE = LINK_RADII.min() * 2 + SAFETY_MARGIN      
LAG_SWEEP = [(10.0, 0.06, 0.45),
             (12.0, 0.08, 0.28),
             (10.0, 0.10, 0.70),
             (12.0, 0.10, 1.00),
             (16.0, 0.15, 1.00)]
MAX_LAG = float(os.environ.get('DUAL_ARM_MAX_LAG', '1.0'))
LEAD_CAP = int(os.environ.get('DUAL_ARM_LEAD_COMBOS', '16'))
BUDGET_S = float(os.environ.get('DUAL_ARM_S24_BUDGET_S', '300'))
VEL_SAFETY = 1.10


def interp(path, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(path) - 1
    if n <= 0: return path[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return path[idx] + a * (path[min(idx + 1, n)] - path[idx])


def mesh_scan(P, names, bases, stride=2, pairs=None):
    """Definitive scan on the REAL MESH at CLEAR_M. `pairs` restricts the scan
    (targeted pre-check of the offending pairs)."""
    if pairs is None:
        pairs = [(names[i], names[j]) for i in range(len(names))
                 for j in range(i + 1, len(names))]
    cs = 0; mind = float('inf')
    for ni, nj in pairs:
        K = min(len(P[ni]), len(P[nj]))
        for k in range(0, K, stride):
            d = pair_min_dist(P[ni][k], bases[ni], P[nj][k], bases[nj])
            if d < mind: mind = d
            if d < CLEAR_M: cs += 1
    return cs, (mind if mind != float('inf') else 0.0)


def lead_combos(coll_pairs, cap=LEAD_CAP):
    combos = [frozenset()]                     
    n = min(len(coll_pairs), 4)                 
    for mask in range(1 << n):
        held = set()
        for i in range(n):
            a, b = coll_pairs[i]
            held.add(b if (mask >> i) & 1 else a)  
        f = frozenset(held)
        if f not in combos:
            combos.append(f)
    return combos[:cap + 1]


def kuramoto(paths, duration, names, bases, k_lag, band, lag_cap,
             held_set=frozenset()):
    d_safe = MIN_SAFE + band
    N = len(names); phi = np.zeros(N); omega = np.full(N, 1.0 / duration)
    hold_steps = int(0.05 * duration / DT) if held_set else 0
    held = np.array([n in held_set for n in names], bool)
    traj = {n: [] for n in names}; nearmiss = 0
    nsteps = int((2.0 + 2.0 * min(lag_cap, 1.0)) * duration / DT)
    steps_done = 0
    for step in range(nsteps):
        q = {n: interp(paths[n], phi[i]) for i, n in enumerate(names)}
        for n in names: traj[n].append(q[n].copy())
        steps_done = step + 1
        caps = {n: _arm_caps(q[n], bases[n]) for n in names}
        pd = {}
        for a in range(N):
            for b in range(a + 1, N):
                d = caps_min_dist(caps[names[a]], caps[names[b]])
                pd[(a, b)] = d
                if MIN_SAFE <= d < d_safe: nearmiss += 1
        if np.all(phi >= 1 - 1e-9): break
        dphi = omega.copy()
        for a in range(N):
            for b in range(a + 1, N):
                if pd[(a, b)] < d_safe:
                    rep = k_lag * ((d_safe - pd[(a, b)]) / d_safe) ** 2
                    if phi[a] < phi[b]: dphi[a] -= rep
                    else:               dphi[b] -= rep
        if step < hold_steps:
            dphi = np.where(held, 0.0, dphi)
        phi = np.clip(phi + DT * np.clip(dphi, 0.0, None), 0, 1)
        nominal = min(1.0, (step + 1) * DT / duration)
        phi = np.maximum(phi, nominal - lag_cap)
    guard = int(1.5 * duration / DT)
    while np.any(phi < 1 - 1e-9) and guard > 0:
        phi = np.clip(phi + DT * omega, 0, 1)
        q = {n: interp(paths[n], phi[i]) for i, n in enumerate(names)}
        for n in names: traj[n].append(q[n].copy())
        steps_done += 1; guard -= 1
    real_dur = max(duration, steps_done * DT)
    return {n: np.array(traj[n]) for n in names}, nearmiss, real_dur


def resample(pos, duration):
    nout = max(2, int(round(duration * RATE_HZ)))
    sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
    return np.clip(np.vstack([np.interp(sout, sin, pos[:, j]) for j in range(NDOF)]).T,
                   POS_LIM[:, 0], POS_LIM[:, 1])


def simultaneity(P, names, duration):
    K = min(len(P[n]) for n in names)
    if K < 3:
        return 1.0
    moving = np.zeros((K - 1, len(names)), bool)
    for i, n in enumerate(names):
        step = np.max(np.abs(np.diff(np.asarray(P[n])[:K], axis=0)), axis=1)
        moving[:, i] = step > 1e-5
    nmov = moving.sum(axis=1)
    active = nmov >= 1
    if not active.any():
        return 1.0
    return float((nmov[active] > 1).mean())


def limit_scale(P, names, duration, compress=False):
    s = 0.0 if compress else 1.0
    for n in names:
        pos = P[n]; dt = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        for j in range(NDOF):
            vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
            if vp > VEL_LIM[j]: s = max(s, vp / VEL_LIM[j])
            if ap > ACC_LIM[j]: s = max(s, float(np.sqrt(ap / ACC_LIM[j])))
    return s


def sweep(paths, duration, names, bases, res_pairs, deadline):
    total_nm = 0; tried = 0
    combos = lead_combos(res_pairs)
    cfgs_sorted = sorted([c for c in LAG_SWEEP if c[2] <= MAX_LAG + 1e-9],
                         key=lambda c: c[2])
    for (kl, bd, cap) in cfgs_sorted:
        for held in combos:
            if time.time() > deadline:
                print(f'  budget reached after {tried} lag run(s)')
                return None, None, None, total_nm, None
            sync, nm, real_dur = kuramoto(paths, duration, names, bases,
                                          kl, bd, cap, held)
            total_nm += nm; tried += 1
            sr = {n: resample(sync[n], real_dur) for n in names}
            cs_r, _ = mesh_scan(sr, names, bases, stride=4, pairs=res_pairs)
            if cs_r > 0:
                continue
            cs, mind = mesh_scan(sr, names, bases, stride=2)
            if cs == 0:
                order = ('free' if not held
                         else 'hold=' + ','.join(sorted(held)))
                return sr, real_dur, (kl, bd, cap, order), total_nm, mind
    print(f'  exhausted {tried} lag run(s) over {len(combos)} ordering(s) '
          f'x {len(cfgs_sorted)} lag cap(s) (max_lag={MAX_LAG})')
    return None, None, None, total_nm, None


def _write(out_path, names, chosen, kdur, base_dur, meta, splines,
           needs_retraction, residual_pairs, label, method, mind):
    out = {'duration': float(kdur), 'arm_names': names,
           'final_duration_s': round(float(kdur), 4),
           'duration_overhead': round(float(kdur) / base_dur - 1, 4),
           'needs_retraction': bool(needs_retraction),
           'residual_pairs': residual_pairs,
           'synchronisation_report': {
               'collision_free': not needs_retraction,
               'min_dist_m': round(float(mind), 4) if mind is not None else None,
               'method': method}}
    for n in names:
        pos = np.asarray(chosen[n]); dt = kdur / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[n] = {'robot_name': n,
                  'metadata': {**meta[n], 'duration': float(kdur)},
                  'spline': splines[n],
                  'trajectory': {'time': np.linspace(0., kdur, len(pos)).tolist(),
                                 'positions': pos.tolist(),
                                 'velocities': vel.tolist(),
                                 'accelerations': acc.tolist(),
                                 'arc_fracs': np.linspace(0., 1., len(pos)).tolist()}}
    with open(out_path, 'w') as fh: json.dump(out, fh)
    print(f'  Saved: {out_path} ({label})')


def main():
    print('\n' + '=' * 66)
    print('  STEP 24  --  Kuramoto-ALONE temporal resolver [6 ARM]')
    print('=' * 66)
    for f, s in [('s22_trajectories.json', 'step_22'),
                 ('s23_collision_map.json', 'step_23')]:
        if not os.path.exists(f): print(f'  {f} not found -- run {s}'); sys.exit(1)
    with open('s22_trajectories.json') as fh: t = json.load(fh)
    with open('s23_collision_map.json') as fh: cmap = json.load(fh)
    names = t.get('arm_names', ARM_NAMES); duration = float(t['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in names}
    paths = {n: np.array(t[n]['trajectory']['positions'], float) for n in names}
    meta = {n: t[n]['metadata'] for n in names}
    splines = {n: t[n].get('spline', {}) for n in names}
    coll_pairs = [(p['arm_i'], p['arm_j']) for p in cmap.get('pairs', [])
                  if p.get('status') == 'COLLISION']
    print(f'\n  Arms: {names}  dur={duration:.2f}s  '
          f'{len(coll_pairs)} colliding pair(s): {coll_pairs}')

    base_P = {n: resample(paths[n], duration) for n in names}
    if not coll_pairs:
        _write('s24_resolved.json', names, base_P, duration, duration, meta,
               splines, False, [], 'no collision', 'none_needed', None)
        print('  No collision. Next : ros2 run dual_arm_sync step_25\n'); return

    t0 = time.time(); deadline = t0 + BUDGET_S
    chosen, kdur, cfg, nm, mind = sweep(paths, duration, names, bases,
                                        coll_pairs, deadline)
    ks = round((time.time() - t0) * 1000, 1)
    if chosen is not None:
        sc = limit_scale(chosen, names, kdur, compress=True)
        if sc > 1e-6 and (sc * VEL_SAFETY < 0.98 or sc > 1.0):
            cand_dur = max(kdur * sc * VEL_SAFETY, 0.5)
            cand = {n: resample(chosen[n], cand_dur) for n in names}
            cs2, mind2 = mesh_scan(cand, names, bases, stride=2)
            if cs2 == 0:
                verb = 'compressed' if cand_dur < kdur else 'stretched'
                print(f'  vel/acc {verb} x{sc:.2f} -> duration '
                      f'{kdur:.2f}s -> {cand_dur:.2f}s (re-verified clear)')
                chosen, kdur, mind = cand, cand_dur, mind2
            elif sc > 1.0:
                kdur = kdur * sc * VEL_SAFETY
                chosen = {n: resample(chosen[n], kdur) for n in names}
                print(f'  vel/acc rescale x{sc:.2f} -> duration {kdur:.2f}s')
        sim = simultaneity(chosen, names, kdur)
        print(f'  KURAMOTO SUCCESS (timing alone): {ks:.0f}ms  cfg={cfg}  '
              f'nearmiss={nm}  min={mind * 100:.1f}cm  dur {duration:.1f}s -> {kdur:.2f}s')
        print(f'  SIMULTANEITY: {100 * sim:.0f}% of the schedule has >1 arm '
              f'moving  (lag_cap={cfg[2]}; 1.0 = take-turns)')
        _write('s24_resolved.json', names, chosen, kdur, duration, meta, splines,
               False, [], 'Kuramoto SUCCESS (timing alone)', 'kuramoto_phase_lag',
               mind)
        print('  Next : ros2 run dual_arm_sync step_25\n'); return

    residual = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c, _ = mesh_scan(base_P, names, bases, stride=2,
                             pairs=[(names[i], names[j])])
            if c > 0:
                residual.append([names[i], names[j]])
    print(f'  Kuramoto alone: {ks:.0f}ms, nearmiss={nm} -- timing cannot separate '
          f'{residual}')
    _write('s24_resolved.json', names, base_P, duration, duration, meta, splines,
           True, residual, f'timing insufficient ({len(residual)} pairs)',
           'kuramoto_insufficient', None)
    print('  Next : ros2 run dual_arm_sync step_25 (retraction + Kuramoto)\n')


if __name__ == '__main__': main()