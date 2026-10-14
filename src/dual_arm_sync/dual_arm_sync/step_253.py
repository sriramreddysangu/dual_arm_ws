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
from _robot8x import (ROBOT_BASES, ARM_NAMES, pair_min_dist,
                      CLEAR_M, PREFILTER_M, _caps_surface)

LAG_GRID = int(os.environ.get('DUAL_ARM_S253_GRID', '60'))
DO_LAGSCAN = os.environ.get('DUAL_ARM_S253_LAGSCAN', '1') != '0'

def coordination_obstacle(pi, pj, bi, bj, G=LAG_GRID):
    """Boolean (G,G) map of U_ij: entry [a,b] is True when arm i at arc a/(G-1)
    collides with arm j at arc b/(G-1). Same gate as the lockstep scan."""
    ia = np.linspace(0, len(pi) - 1, G).astype(int)
    ib = np.linspace(0, len(pj) - 1, G).astype(int)
    QA = pi[ia]; QB = pj[ib]
    obst = np.zeros((G, G), bool)
    for a in range(G):
        for b in range(G):
            if _caps_surface(QA[a], bi, QB[b], bj) > PREFILTER_M:
                continue
            if pair_min_dist(QA[a], bi, QB[b], bj) < CLEAR_M:
                obst[a, b] = True
    return obst


def monotone_reachable(obst):
    G = len(obst)
    if obst[0, 0] or obst[G - 1, G - 1]:
        return False
    R = np.zeros((G, G), bool); R[0, 0] = True
    for a in range(G):
        for b in range(G):
            if (a == 0 and b == 0) or obst[a, b]:
                continue
            up = R[a - 1, b] if a else False
            left = R[a, b - 1] if b else False
            diag = R[a - 1, b - 1] if (a and b) else False
            R[a, b] = bool(up or left or diag)
    return bool(R[G - 1, G - 1])


def min_constant_lag(obst):
    G = len(obst)
    for dc in range(0, G):
        for lead in (0, 1):
            ok = True
            for k in range(G + dc):
                iL = min(k, G - 1)
                iLag = min(max(k - dc, 0), G - 1)
                a, b = (iLag, iL) if lead else (iL, iLag)
                if obst[a, b]:
                    ok = False
                    break
            if ok:
                return dc / (G - 1.0), ('j_leads' if lead else 'i_leads')
    return None, None


def timing_feasibility(pi, pj, bi, bj, ni, nj):
    t0 = time.time()
    obst = coordination_obstacle(pi, pj, bi, bj)
    lam, who = min_constant_lag(obst)
    mono = monotone_reachable(obst)
    rec = {
        'grid': LAG_GRID,
        'interference_pct': round(100.0 * float(obst.mean()), 2),
        'constant_lag_clears': lam is not None,
        'min_lambda': None if lam is None else round(float(lam), 4),
        'lag_leader': (None if who is None else (ni if who == 'i_leads' else nj)),
        'monotone_schedule_exists': bool(mono),
        'verdict': 'KURAMOTO_ALONE' if mono else 'NEEDS_DEFORMATION',
        'scan_time_s': round(time.time() - t0, 2)}
    return rec

def main():
    with open('s252_trajectories.json') as fh:
        t = json.load(fh)
    arm_names = t.get('arm_names', ARM_NAMES)
    pos = {n: np.array(t[n]['trajectory']['positions'], float) for n in arm_names}
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    N = len(arm_names)
    results = []
    print('\n  STEP 253: real-mesh scan, {} arm(s), {} pair(s)'.format(
        N, N * (N - 1) // 2))

    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = arm_names[i], arm_names[j]
            K = min(len(pos[ni]), len(pos[nj]))
            arc = np.linspace(0., 1., K)
            first = -1; n_coll = 0; dmin = float('inf'); worst_k = -1
            for k in range(K):
                if _caps_surface(pos[ni][k], bases[ni],
                                 pos[nj][k], bases[nj]) > PREFILTER_M:
                    continue
                d = pair_min_dist(pos[ni][k], bases[ni], pos[nj][k], bases[nj])
                if d < dmin:
                    dmin = d; worst_k = k
                if d < CLEAR_M:
                    n_coll += 1
                    if first < 0:
                        first = k
            rec = {'pair': '{}<->{}'.format(ni, nj), 'arm_i': ni, 'arm_j': nj,
                   'status': 'COLLISION' if n_coll > 0 else 'SAFE',
                   'steps_checked': K, 'coll_steps': n_coll,
                   'collision_fraction': round(n_coll / K, 5) if K else 0.0,
                   'global_min_dist_m': round(float(min(dmin, 1.0)), 4),
                   'closest_arc': (round(float(arc[worst_k]), 5)
                                   if worst_k >= 0 else None),
                   'first_collision_arc': (round(float(arc[first]), 5)
                                           if first >= 0 else None),
                   'per_timestep_detail': []}
            if n_coll > 0 and DO_LAGSCAN:
                print('    {}<->{}: COLLISION ({} steps, min {:.1f} cm) '
                      '-- gridding U_ij...'.format(
                          ni, nj, n_coll, rec['global_min_dist_m'] * 100))
                rec['timing_feasibility'] = timing_feasibility(
                    pos[ni], pos[nj], bases[ni], bases[nj], ni, nj)
            results.append(rec)

    n_coll_pairs = sum(1 for r in results if r['status'] == 'COLLISION')
    out = {'overall_status': 'COLLISION' if n_coll_pairs else 'SAFE',
           'n_pairs': len(results), 'n_colliding_pairs': n_coll_pairs,
           'clear_m': CLEAR_M, 'pairs': results}
    with open('s253_collision_map.json', 'w') as fh:
        json.dump(out, fh)
    bar = '=' * 78
    print('\n' + bar)
    print('  PAIRWISE RESULT  (naive seed: all arms start together)')
    print(bar)
    print('  {:<18} {:>9} {:>9}  {}'.format(
        'pair', 'min[cm]', 'coll[%]', 'mechanism required'))
    print('  ' + '-' * 74)
    for r in sorted(results, key=lambda x: (x['status'] != 'COLLISION',
                                            x['global_min_dist_m'])):
        tag = '{}<->{}'.format(r['arm_i'], r['arm_j'])
        if r['status'] != 'COLLISION':
            print('  {:<18} {:>9.2f} {:>9}  clear'.format(
                tag, r['global_min_dist_m'] * 100, '-'))
            continue
        tf = r.get('timing_feasibility')
        if tf is None:
            note = 'collides (lag scan disabled)'
        elif tf['verdict'] == 'KURAMOTO_ALONE':
            if tf['constant_lag_clears']:
                note = ('KURAMOTO alone -- constant lag lambda={:.3f}, {} leads '
                        '({:.1f}% of U_ij blocked)'.format(
                            tf['min_lambda'], tf['lag_leader'],
                            tf['interference_pct']))
            else:
                note = ('KURAMOTO alone -- needs a NON-constant schedule '
                        '({:.1f}% of U_ij blocked)'.format(tf['interference_pct']))
        else:
            note = ('DEFORMATION -- no monotone schedule exists '
                    '({:.1f}% of U_ij blocked)'.format(tf['interference_pct']))
        print('  {:<18} {:>9.2f} {:>9.2f}  {}'.format(
            tag, r['global_min_dist_m'] * 100, 100 * r['collision_fraction'], note))
    print('  ' + '-' * 74)

    kur = [r for r in results if r.get('timing_feasibility', {})
           .get('verdict') == 'KURAMOTO_ALONE']
    dfm = [r for r in results if r.get('timing_feasibility', {})
           .get('verdict') == 'NEEDS_DEFORMATION']
    print('  {}/{} pairs collide: {} resolvable by timing, {} need deformation.'
          .format(n_coll_pairs, len(results), len(kur), len(dfm)))
    if kur and dfm:
        print('  This configuration exercises BOTH mechanisms -- good demo material.')
        print('  Kuramoto-alone pair(s) : {}'.format(
            ', '.join(r['pair'] for r in kur)))
        print('  Deformation pair(s)    : {}'.format(
            ', '.join(r['pair'] for r in dfm)))
        print('  Dual-arm replays of those pairs:')
        for r in kur + dfm:
            print('    python3 step_200.py --replay-pair {},{}'.format(
                r['arm_i'], r['arm_j']))
    print(bar)
    print('  Saved: s253_collision_map.json   Next: step_254')


if __name__ == '__main__':
    main()