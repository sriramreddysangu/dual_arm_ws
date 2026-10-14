#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'nine'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, os, time
import numpy as np
from _robot8x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ,
                      ROBOT_BASES, ARM_NAMES, LINK_RADII, SAFETY_MARGIN,
                      _arm_caps, caps_min_dist, CLEAR_M, SCAN_STRIDE,
                      mesh_scan, residual_of, resample, interp, limit_scale,
                      simultaneity)

DT = 0.01
MIN_SAFE = LINK_RADII.min() * 2 + SAFETY_MARGIN
MAX_LAG = float(os.environ.get('DUAL_ARM_MAX_LAG', '1.0'))
LEAD_CAP = int(os.environ.get('DUAL_ARM_LEAD_COMBOS', '16'))
S24_BUDGET = float(os.environ.get('DUAL_ARM_S24_BUDGET_S', '300'))
VEL_SAFETY = 1.10
CREEP_MIN = float(os.environ.get('DUAL_ARM_CREEP_MIN', '0.0'))
BRAKE_CFGS = [(12.0, 0.11), (10.0, 0.09), (16.0, 0.16)]
LAG_CAPS = [float(x) for x in os.environ.get(
    'DUAL_ARM_LAG_CAPS', '0.05,0.10,0.15,0.22,0.30,0.45,0.70,1.00').split(',')]
LAG_SWEEP = [(k, b, c) for c in LAG_CAPS for (k, b) in BRAKE_CFGS]


def lead_combos(coll_pairs, cap=LEAD_CAP):
    combos = [frozenset()]
    n = min(len(coll_pairs), 4)
    for mask in range(1 << n):
        held = set()
        for i in range(n):
            a, b = coll_pairs[i]
            held.add(b if (mask >> i) & 1 else a)
        f = frozenset(held)
        if f not in combos: combos.append(f)
    return combos[:cap + 1]


def kuramoto(paths, duration, names, bases, k_lag, band, lag_cap,
             held_set=frozenset(), conflict=None):
    d_safe = CLEAR_M + band
    d_hard = CLEAR_M
    N = len(names); phi = np.zeros(N); omega = np.full(N, 1.0 / duration)
    hold_steps = int(0.05 * duration / DT) if held_set else 0
    held = np.array([n in held_set for n in names], bool)
    _ix = {n: i for i, n in enumerate(names)}
    act = None
    if conflict:
        act = {(min(_ix[a], _ix[b]), max(_ix[a], _ix[b]))
               for a, b in conflict if a in _ix and b in _ix}
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
                if act is not None and (a, b) not in act: continue
                d = caps_min_dist(caps[names[a]], caps[names[b]])
                pd[(a, b)] = d
                if d_hard <= d < d_safe: nearmiss += 1
        if np.all(phi >= 1 - 1e-9): break
        scale = np.ones(N)
        for (a, b), d in pd.items():
            if d < d_safe:
                x = (d_safe - d) / max(d_safe - d_hard, 1e-6)
                s = 0.0 if d < d_hard else float(np.clip(1.0 - x, CREEP_MIN, 1.0))
                t = a if phi[a] < phi[b] else b      
                scale[t] = min(scale[t], s)
        dphi = omega * scale
        if step < hold_steps:
            dphi = np.where(held, 0.0, dphi)
        phi = np.clip(phi + DT * dphi, 0, 1)
        nominal = (step + 1) * DT / duration           
        phi = np.maximum(phi, min(1.0, nominal - lag_cap))
        phi = np.clip(phi, 0, 1)
    guard = int(1.5 * duration / DT)
    while np.any(phi < 1 - 1e-9) and guard > 0:
        phi = np.clip(phi + DT * omega, 0, 1)
        q = {n: interp(paths[n], phi[i]) for i, n in enumerate(names)}
        for n in names: traj[n].append(q[n].copy())
        steps_done += 1; guard -= 1
    real_dur = max(duration, steps_done * DT)
    return {n: np.array(traj[n]) for n in names}, nearmiss, real_dur


def kuramoto_sweep(paths, duration, names, bases, res_pairs, deadline,
                   want_nm=False):
    total_nm = 0
    combos = lead_combos(res_pairs or [])
    cfgs = sorted([c for c in LAG_SWEEP if c[2] <= MAX_LAG + 1e-9],
                  key=lambda c: c[2])
    for (kl, bd, cap) in cfgs:
        for held in combos:
            if time.time() > deadline:
                return (None, None, None, total_nm, None) if want_nm else (None, None, None)
            sync, nm, real_dur = kuramoto(paths, duration, names, bases,
                                          kl, bd, cap, held, conflict=res_pairs)
            total_nm += nm
            sr = {n: resample(sync[n], real_dur) for n in names}
            cs_r, _ = mesh_scan(sr, names, bases, stride=SCAN_STRIDE,
                                pairs=res_pairs)
            if cs_r > 0: continue
            cs, mind = mesh_scan(sr, names, bases, stride=SCAN_STRIDE)
            if cs == 0:
                order = 'free' if not held else 'hold=' + ','.join(sorted(held))
                cfg = (kl, bd, cap, order)
                return (sr, real_dur, cfg, total_nm, mind) if want_nm else (sr, real_dur, cfg)
    return (None, None, None, total_nm, None) if want_nm else (None, None, None)


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
        out[n] = {'robot_name': n, 'metadata': {**meta[n], 'duration': float(kdur)},
                  'spline': splines[n],
                  'trajectory': {'time': np.linspace(0., kdur, len(pos)).tolist(),
                                 'positions': pos.tolist(), 'velocities': vel.tolist(),
                                 'accelerations': acc.tolist(),
                                 'arc_fracs': np.linspace(0., 1., len(pos)).tolist()}}
    with open(out_path, 'w') as fh: json.dump(out, fh)
    print(f'  Saved: {out_path} (9-arm)')

def main():
    with open('s132_trajectories.json') as fh: t = json.load(fh)
    with open('s133_collision_map.json') as fh: cmap = json.load(fh)
    names = t.get('arm_names', ARM_NAMES); duration = float(t['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in names}
    paths = {n: np.array(t[n]['trajectory']['positions'], float) for n in names}
    meta = {n: t[n]['metadata'] for n in names}
    splines = {n: t[n].get('spline', {}) for n in names}
    coll_pairs = [(p['arm_i'], p['arm_j']) for p in cmap.get('pairs', [])
                  if p.get('status') == 'COLLISION']
    base_P = {n: resample(paths[n], duration) for n in names}
    if not coll_pairs:
        _write('s134_resolved.json', names, base_P, duration, duration, meta,
               splines, False, [], 'no collision', 'none_needed', None)
        return
    deadline = time.time() + S24_BUDGET
    chosen, kdur, cfg, nm, mind = kuramoto_sweep(paths, duration, names, bases,
                                                 coll_pairs, deadline, want_nm=True)
    if chosen is not None:
        sc = limit_scale(chosen, names, kdur, compress=True)
        if sc > 1e-6 and (sc * VEL_SAFETY < 0.98 or sc > 1.0):
            cand_dur = max(kdur * sc * VEL_SAFETY, 0.5)
            cand = {n: resample(chosen[n], cand_dur) for n in names}
            cs2, mind2 = mesh_scan(cand, names, bases)
            if cs2 == 0:
                chosen, kdur, mind = cand, cand_dur, mind2
            elif sc > 1.0:
                kdur = kdur * sc * VEL_SAFETY
                chosen = {n: resample(chosen[n], kdur) for n in names}
        print(f'  STEP 134 KURAMOTO SUCCESS (timing alone) cfg={cfg} '
              f'min={mind*100:.1f}cm dur {duration:.1f}s -> {kdur:.2f}s')
        print(f'  SIMULTANEITY: {100*simultaneity(chosen, names):.0f}% '
              f'(lag_cap={cfg[2]}; 1.0 = take-turns)')
        _write('s134_resolved.json', names, chosen, kdur, duration, meta, splines,
               False, [], 'Kuramoto SUCCESS (timing alone)', 'kuramoto_phase_lag', mind)
        return
    residual = residual_of(base_P, names, bases)
    print(f'  STEP 134: timing cannot separate {residual} -> step 135')
    _write('s134_resolved.json', names, base_P, duration, duration, meta, splines,
           True, [list(p) for p in residual],
           f'timing insufficient ({len(residual)} pairs)', 'kuramoto_insufficient', None)


if __name__ == '__main__':
    main()