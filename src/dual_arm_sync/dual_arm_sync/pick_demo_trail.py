#!/usr/bin/env python3

import sys, json, itertools
from typing import Dict, List
import numpy as np


_PI, _PI_2 = np.pi, np.pi / 2.0
L1, L2, L3, L4, A = 0.1525, 0.6200, 0.5590, 0.1210, 0.0345

DH = np.array([
    [0.0,    0.0,  0.0,    L1],
    [-_PI_2, 0.0, -_PI_2,  A ],
    [0.0,    L2,   _PI_2,  0.0],
    [_PI_2,  0.0,  0.0,    L3],
    [-_PI_2, 0.0,  0.0,    0.0],
    [_PI_2,  0.0,  0.0,    L4],
], dtype=float)

POS_LIM = np.array([
    [-2*_PI, 2*_PI], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI, 2*_PI], [-2*_PI, 2*_PI], [-2*_PI, 2*_PI],
], dtype=float)

NDOF = 6
ROBOT_BASES = {
    'dsr01': np.array([ 0.00,  0.695, 0.0]),
    'dsr02': np.array([ 0.00, -0.695, 0.0]),
    'dsr03': np.array([ 1.39,  0.695, 0.0]),
    'dsr04': np.array([ 1.39, -0.695, 0.0]),
    'dsr05': np.array([-1.39,  0.695, 0.0]),
    'dsr06': np.array([-1.39, -0.695, 0.0]),
}
ARMS = ['dsr01', 'dsr02', 'dsr03', 'dsr04', 'dsr05', 'dsr06']

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12
RADII_THR     = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN

PATH_STEPS = 201    
LAG_GRID   = 121     

def link_origins(q, base):
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o


def sample_line(qs, qt, m):
    u = np.linspace(0.0, 1.0, m)
    cfg = qs[None, :] + u[:, None] * (qt - qs)[None, :]
    return np.clip(cfg, POS_LIM[:, 0], POS_LIM[:, 1])


def origins_along(qs, qt, base, m):
    return np.stack([link_origins(q, base) for q in sample_line(qs, qt, m)], axis=0)


def pair_collides(Oa, Ob):
    d = np.linalg.norm(Oa[:, :, None, :] - Ob[:, None, :, :], axis=3)  
    hit = (d < RADII_THR[None, :, :]).reshape(len(d), -1).any(axis=1)
    return bool(hit.any()), float(d.min()), float(hit.mean())


def coordination_matrix(Oa, Ob):
    d = np.linalg.norm(Oa[:, None, :, None, :] - Ob[None, :, None, :, :], axis=4)
    return (d < RADII_THR[None, None, :, :]).any(axis=(2, 3))


def lag_clears(C):
    m = C.shape[0]
    best = None
    for k in range(-(m - 1), m):
        t = np.arange(min(0, k), max(m - 1, m - 1 + k) + 1)
        if not C[np.clip(t - k, 0, m - 1), np.clip(t, 0, m - 1)].any():
            lam = abs(k) / (m - 1.0)
            if best is None or lam < best:
                best = lam
    return best


def analyse_trial(rec):
    sj, tj = rec.get('start_joints') or {}, rec.get('target_joints') or {}
    if not all(n in sj and n in tj for n in ARMS):
        return None
    starts  = {n: np.array(sj[n], dtype=float) for n in ARMS}
    targets = {n: np.array(tj[n], dtype=float) for n in ARMS}

    Opath = {n: origins_along(starts[n], targets[n], ROBOT_BASES[n], PATH_STEPS)
             for n in ARMS}

    pairs = []
    for a, b in itertools.combinations(ARMS, 2):
        hit, dmin, frac = pair_collides(Opath[a], Opath[b])
        if not hit:
            pairs.append({'pair': (a, b), 'collision': False,
                          'min_cm': round(dmin * 100, 2)})
            continue
        Ca = origins_along(starts[a], targets[a], ROBOT_BASES[a], LAG_GRID)
        Cb = origins_along(starts[b], targets[b], ROBOT_BASES[b], LAG_GRID)
        C  = coordination_matrix(Ca, Cb)
        lam = lag_clears(C)
        pairs.append({
            'pair': (a, b), 'collision': True,
            'min_cm': round(dmin * 100, 2),
            'coll_pct': round(100 * frac, 1),
            'interf_pct': round(100 * float(C.mean()), 1),
            'lag_clears': lam is not None,
            'min_lambda': None if lam is None else round(lam, 3),
            'verdict': 'LAG-CLEARS' if lam is not None else 'NEEDS-DEFORM',
        })

    coll = [p for p in pairs if p['collision']]
    n_lag  = sum(1 for p in coll if p['lag_clears'])
    n_def  = sum(1 for p in coll if not p['lag_clears'])
    return {
        'trial': rec.get('trial'),
        'category': rec.get('category'),
        'executed': bool(rec.get('executed')),
        'exec_success': bool(rec.get('exec_success')),
        'pairs': pairs,
        'n_coll': len(coll),
        'n_lag': n_lag,
        'n_deform': n_def,
        'both': n_lag >= 1 and n_def >= 1,
        'tightest_cm': min((p['min_cm'] for p in coll), default=None),
    }


def arg_val(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def print_trial(res, verbose=True):
    print('  trial {:>4}  [{}]  exec_success={}  collisions={} '
          '(lag={}, deform={})  tightest={} cm'.format(
              res['trial'], res['category'], res['exec_success'],
              res['n_coll'], res['n_lag'], res['n_deform'], res['tightest_cm']))
    if not verbose:
        return
    for p in res['pairs']:
        if not p['collision']:
            continue
        lam = '-' if p['min_lambda'] is None else '{:.3f}'.format(p['min_lambda'])
        print('      {}<->{}  min {:6.2f} cm  path {:5.1f}%  U_ij {:5.1f}%  '
              'lambda {:>6}  {}'.format(
                  p['pair'][0], p['pair'][1], p['min_cm'], p['coll_pct'],
                  p['interf_pct'], lam, p['verdict']))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    top = int(arg_val('--top', '5'))
    emit = arg_val('--emit')
    only_exec = '--only-executed' in sys.argv

    with open(path, 'r') as f:
        data = json.load(f)
    trials = data['trials'] if 'trials' in data else [data]
    print('\n  Loaded {} trials from {}'.format(len(trials), path))

    if emit is not None:
        for t in trials:
            if int(t.get('trial', -1)) == int(emit):
                out = {
                    'trial': t.get('trial'),
                    'category': t.get('category'),
                    'start_joints': t.get('start_joints'),
                    'target_joints': t.get('target_joints'),
                }
                name = 'demo_trial_{}.json'.format(emit)
                with open(name, 'w') as f:
                    json.dump(out, f, indent=2)
                print('  wrote {}'.format(name))
                res = analyse_trial(t)
                if res:
                    print()
                    print_trial(res)
                    print('\n  Six-arm clip:')
                    print('    python3 raw_six_move.py --from-json {}'.format(name))
                    print('  Dual-arm replays:')
                    for p in res['pairs']:
                        if p['collision']:
                            print('    python3 raw_six_move.py --arms {},{} '
                                  '--from-json {}   # {}'.format(
                                      p['pair'][0], p['pair'][1], name,
                                      p['verdict']))
                return
        raise SystemExit('  trial {} not found'.format(emit))

    results = []
    for i, t in enumerate(trials):
        if only_exec and not t.get('executed'):
            continue
        r = analyse_trial(t)
        if r is not None:
            results.append(r)
        if (i + 1) % 25 == 0:
            print('    ...scanned {}/{}'.format(i + 1, len(trials)))

    both = [r for r in results if r['both']]
    print('\n  {} trials analysable, {} contain BOTH a lag-clearable and a '
          'deformation-only collision.'.format(len(results), len(both)))

    
    both.sort(key=lambda r: (-(r['n_lag'] >= 1 and r['n_deform'] >= 1),
                             -r['n_coll'],
                             -int(r['exec_success']),
                             r['tightest_cm'] if r['tightest_cm'] is not None else 99))

    print('\n' + '=' * 78)
    print('  BEST CANDIDATES FOR THE SIX-ARM VIDEO')
    print('=' * 78)
    for r in both[:top]:
        print_trial(r)
        print()

    if not both:
        print('  No single trial has both kinds. Fallbacks:')
        multi = sorted([r for r in results if r['n_coll'] >= 2],
                       key=lambda r: -r['n_coll'])[:top]
        print('  Trials with >=2 colliding pairs (all same kind):')
        for r in multi:
            print_trial(r, verbose=False)
        print('\n  Options: relax SAFETY_MARGIN, or stage the clip from two '
              'trials, or hand-build a config (raw_six_move.py prompts).')
    else:
        best = both[0]
        print('  Next:')
        print('    python3 pick_demo_trial.py {} --emit {}'.format(
            path, best['trial']))
    print()


if __name__ == '__main__':
    main()