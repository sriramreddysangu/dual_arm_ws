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
from _robot8x import (ROBOT_BASES, eval_cp, mesh_scan, residual_of,
                      resample, limit_scale, simultaneity)
import step_104 as S104
import step_105 as S105

OUT_PATH = 's255_synchronized.json'


def _write(names, chosen, kdur, base_dur, meta, resolved, method, extra=None):
    """Same record step_105 writes, under the 200-series filename."""
    out = {'duration': float(kdur), 'arm_names': names,
           'final_duration_s': round(float(kdur), 4),
           'duration_overhead': round(float(kdur) / base_dur - 1, 4),
           'synchronisation_report': {'collision_free': resolved,
                                      'method': method},
           'final_verification': {'verified_collision_free': resolved}}
    out.update(extra or {})
    for n in names:
        pos = np.asarray(chosen[n])
        dt = kdur / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        out[n] = {'robot_name': n,
                  'metadata': {**meta[n], 'duration': float(kdur)},
                  'trajectory': {'time': np.linspace(0., kdur, len(pos)).tolist(),
                                 'positions': pos.tolist(),
                                 'velocities': vel.tolist(),
                                 'accelerations': acc.tolist(),
                                 'arc_fracs': np.linspace(0., 1., len(pos)).tolist()}}
    with open(OUT_PATH, 'w') as fh:
        json.dump(out, fh)
    print('  Saved: {}'.format(OUT_PATH))

def _arc(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    c = np.concatenate([[0.0], np.cumsum(d)])
    if c[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(P))
    return c / c[-1]


def _on_arc(P, s_out):
    s_in = _arc(P)
    return np.vstack([np.interp(s_out, s_in, P[:, j])
                      for j in range(P.shape[1])]).T


def _box(X, w):
    if w < 3:
        return X.copy()
    if w % 2 == 0:
        w += 1
    h = w // 2
    Y = X.copy()
    n = len(X)
    for i in range(n):
        Y[i] = X[max(0, i - h):min(n, i + h + 1)].mean(axis=0)
    Y[0] = X[0]
    Y[-1] = X[-1]
    for i in range(1, min(h, n // 4)):      
        f = i / float(h)
        Y[i] = (1 - f) * X[i] + f * Y[i]
        Y[-1 - i] = (1 - f) * X[-1 - i] + f * Y[-1 - i]
    return Y


def minimise_deformation(seed_P, best, names, bases, duration, narc=400):
    window = max(9, narc // 4)       
    grid = np.linspace(0.0, 1.0, narc) 
    s_best = {n: _arc(best[n]) for n in names}
    seed_a = {n: _on_arc(seed_P[n], grid) for n in names}
    delta = {n: _on_arc(best[n], grid) - seed_a[n] for n in names}

    def build(scale):
        out = {}
        for n in names:
            arc_new = seed_a[n] + _box(delta[n], window) * scale
            Q = np.vstack([np.interp(s_best[n], grid, arc_new[:, j])
                           for j in range(arc_new.shape[1])]).T
            Q[0] = best[n][0]
            Q[-1] = best[n][-1]      
            out[n] = Q
        return out

    def clears(P):
        cs, mind = mesh_scan(P, names, bases)
        return cs == 0, mind

    ok0, d0 = clears(best)
    if not ok0:
        return best, None             

    cand = build(1.0)
    ok, dm = clears(cand)
    if not ok:
        print('  minimality: smoothing alone breaks the gate -- keeping the '
              'sharp path')
        return best, None
    keep, keep_d, keep_s = cand, dm, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(6):                
        mid = 0.5 * (lo + hi)
        c = build(mid)
        okm, dmm = clears(c)
        if okm:
            hi = mid
            keep, keep_d, keep_s = c, dmm, mid
        else:
            lo = mid
    return keep, {'window': window, 'scale': round(float(keep_s), 4),
                  'min_dist_before_m': round(float(d0), 4),
                  'min_dist_after_m': round(float(keep_d), 4)}


def main():
    with open('s254_resolved.json') as fh:
        s24 = json.load(fh)
    names = s24['arm_names']
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in names}

    with open('s252_trajectories.json') as fh:
        s22 = json.load(fh)
    duration0 = float(s22['duration'])
    cp0 = {n: S105.global_cp_from(s22[n]) for n in names}
    try:
        with open('s253_collision_map.json') as fh:
            s23 = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        s23 = {}
    coll_pairs_diag = [(p['arm_i'], p['arm_j']) for p in s23.get('pairs', [])
                       if p.get('status') == 'COLLISION']

    diag_fn = getattr(S105, 'escape_alone_diagnostic', None)
    diag_budget = float(getattr(S105, 'ESCAPE_DIAG_BUDGET', 60.0))
    escape_diag = []
    if coll_pairs_diag and diag_fn is not None:
        escape_diag = diag_fn(cp0, names, bases, duration0, coll_pairs_diag,
                              time.time() + diag_budget)
    if escape_diag:
        n_also = sum(1 for e in escape_diag if e.get('escape_alone_resolves'))
        print('  DIAGNOSTIC (escape-alone vs Kuramoto-alone): {}/{} colliding '
              'pair(s) would ALSO be resolvable by escape alone.'.format(
                  n_also, len(escape_diag)))
        for e in escape_diag:
            print('    {}: escape_alone_resolves={}'.format(
                e['pair'], e.get('escape_alone_resolves')))

    if not s24.get('needs_retraction', False):
        out = dict(s24)
        out['final_verification'] = {'verified_collision_free': True,
                                     'reason': 'kuramoto_alone_in_step_254'}
        out['escape_alone_diagnostic'] = escape_diag
        with open(OUT_PATH, 'w') as fh:
            json.dump(out, fh)
        print('  STEP 255: step_254 resolved everything by timing -> passthrough.')
        print('  Saved: {}   Next: step_256'.format(OUT_PATH))
        return

    duration = duration0
    cp = {n: cp0[n].copy() for n in names}
    starts = {n: np.array(s22[n]['metadata']['start_joints'], float) for n in names}
    targets = {n: np.array(s22[n]['metadata']['end_joints'], float) for n in names}
    meta = {n: s22[n]['metadata'] for n in names}
    residual0 = [tuple(p) for p in s24.get('residual_pairs', [])]
    print('  STEP 255: timing-insufficient pairs {}'.format(residual0))

    deadline = time.time() + S105.S25_BUDGET
    t0 = time.time()
    best = None; best_cs = 10**9; best_dur = duration; cons = []
    rnd = 0; stale = 0; last_cs = None
    while time.time() < deadline:
        rnd += 1
        paths = {n: eval_cp(cp[n], duration) for n in names}
        residual = residual_of(paths, names, bases)
        if not residual:
            cs, _ = mesh_scan(paths, names, bases)
            if cs == 0:
                best, best_cs, best_dur = paths, 0, duration
                break
        for arms, prs in S105.clusters_of(residual):
            cp, gcs = S105.escape_repair(cp, arms, prs, bases, names,
                                         duration, deadline)
            tag = 'JOINTLY' if len(arms) > 2 else ''
            print('  round {}: cluster {} ({} pairs) escaped {} -> global_cs~{} '
                  '({:.0f}s)'.format(rnd, arms, len(prs), tag, gcs,
                                     time.time() - t0))
        paths = {n: eval_cp(cp[n], duration) for n in names}
        residual = residual_of(paths, names, bases)
        print('  round {}: after escape residual={}'.format(rnd, residual))
        chosen, kdur, cfg = S104.kuramoto_sweep(paths, duration, names, bases,
                                                residual or None, deadline)
        if chosen is not None:
            best, best_cs, best_dur = chosen, 0, kdur
            print('  KURAMOTO re-iteration SUCCESS cfg={}'.format(cfg))
            break
        cs, _ = mesh_scan(paths, names, bases)
        if cs == 0:
            best, best_cs, best_dur = paths, 0, duration
            break
        if cs < best_cs:
            best, best_cs, best_dur = paths, cs, duration
        if rnd >= 2 and residual and time.time() < deadline:
            cons = S105.stage_C(cp, names, bases, starts, targets, residual,
                                duration, deadline)
            paths = {n: eval_cp(cp[n], duration) for n in names}
            residual = residual_of(paths, names, bases)
            chosen, kdur, cfg = S104.kuramoto_sweep(paths, duration, names,
                                                    bases, residual or None,
                                                    deadline)
            if chosen is not None:
                best, best_cs, best_dur = chosen, 0, kdur
                print('  KURAMOTO after parked-pose clearance SUCCESS cfg={}'
                      .format(cfg))
                break
            base_P = {n: resample(paths[n], duration) for n in names}
            seq, sdur, K = S105.topo_sequential(base_P, names, bases, cons,
                                                duration, residual or None)
            if seq is not None:
                best, best_cs, best_dur = seq, 0, sdur
                print('  TOPOLOGICAL SEQUENTIAL SUCCESS (K={} phases)'.format(K))
                break
        if last_cs is not None and best_cs >= last_cs:
            stale += 1
            if stale >= S105.NO_PROGRESS_STOP:
                print('  no progress in {} rounds -> stopping early '
                      '(structurally unresolvable; {} coll steps)'.format(
                          stale, best_cs))
                break
        else:
            stale = 0
        last_cs = best_cs
        if rnd >= 8:
            break
    if best is not None and best_cs == 0:
        seed_P = {n: eval_cp(cp0[n], duration) for n in names}
        bases_m = {n: np.array(ROBOT_BASES[n], float) for n in names}
        best2, minfo = minimise_deformation(seed_P, best, names, bases_m,
                                            duration)
        if minfo is not None:
            best = best2
            print('  minimality: deformation shrunk to {:.0%} (window {}), '
                  'clearance {:.2f} -> {:.2f} cm'.format(
                      minfo['scale'], minfo['window'],
                      minfo['min_dist_before_m'] * 100,
                      minfo['min_dist_after_m'] * 100))
        else:
            minfo = {}
    else:
        minfo = {}

    dev = {}
    grid_d = np.linspace(0.0, 1.0, 400)
    for n in names:
        opt = _on_arc(eval_cp(cp0[n], duration), grid_d)
        cur = _on_arc(best[n] if best is not None else eval_cp(cp[n], duration),
                      grid_d)
        nrm = np.linalg.norm(cur - opt, axis=1)
        if float(nrm.max()) > 1e-6:
            dev[n] = (round(float(nrm.mean()), 4), round(float(nrm.max()), 4))
    print('  deviation from the seed path (mean, peak) rad: {}'.format(
        dev if dev else 'none'))

    resolved = best_cs == 0 and best is not None
    if best is None:
        best = {n: eval_cp(cp[n], duration) for n in names}
        best_dur = duration
    sc = limit_scale(best, names, best_dur)
    if sc > 1.0:
        best_dur = best_dur * sc * S105.VEL_SAFETY
        best = {n: resample(best[n], best_dur) for n in names}
    if resolved:
        print('  SIMULTANEITY: {:.0f}%'.format(100 * simultaneity(best, names)))
    print('  total resolver time: {:.0f}s (budget {:.0f}s)'.format(
        time.time() - t0, S105.S25_BUDGET))

    method = 'kuramoto_plus_witness_escape' if resolved else 'unresolved'
    _write(names, best, best_dur, duration, meta, resolved, method,
           extra={'escape_alone_diagnostic': escape_diag,
                  'path_deviation_rad': {k: list(v) for k, v in dev.items()},
                  'deformation_minimality': minfo,
                  'residual_pairs_in': [list(p) for p in residual0]})
    if resolved:
        print('  Next: step_256')
    else:
        print('  UNRESOLVED -- step_256 pre-flight will refuse to execute this.')
        print('  Retype a target in step_251, or raise DUAL_ARM_TIME_BUDGET_S.')


if __name__ == '__main__':
    main()