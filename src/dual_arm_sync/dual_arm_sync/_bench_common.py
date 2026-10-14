#!/usr/bin/env python3

import contextlib, io, json, os, shutil, sys, tempfile, time
import numpy as np

CAT = ['NO_COLLISION', 'KURAMOTO', 'RETRACTION', 'UNRESOLVED', 'IK_FAIL']


def quiet(fn, *a, **k):
    
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(*a, **k)
        return True, buf.getvalue()
    except SystemExit:
        return False, buf.getvalue()
    except Exception as e:
        return False, buf.getvalue() + f'\nEXC: {type(e).__name__}: {e}'


def sample_target(rng, R, base, start_q, reach_lo=0.45, reach_hi=0.92, zlo=0.50, zhi=0.90,
                  box=None):
    
    if box is None:                       
        box = ((-0.35, 0.45), (-0.50, 0.50), (zlo, zhi))
    (xlo, xhi), (ylo, yhi), (zl, zh) = box
    for _ in range(400):
        ee = np.array([rng.uniform(xlo, xhi), rng.uniform(ylo, yhi), rng.uniform(zl, zh)])
        r = float(np.linalg.norm(ee[:2] - np.asarray(base)[:2]))
        if reach_lo <= r <= reach_hi:
            return ee
    return R.fk_world(start_q, base)


def scan_all(R, names, bases, P, clear=0.05, stride=2):
    worst = 9.0; bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            K = min(len(P[ni]), len(P[nj]))
            mn = min(R.pair_min_dist(P[ni][k], bases[ni], P[nj][k], bases[nj])
                     for k in range(0, K, stride))
            if mn < clear: bad.append(f'{ni}-{nj}')
            worst = min(worst, mn)
    return bad, worst


def classify(scan_json, kur_json, ret_json, kur_key='needs_retraction'):
    """Map the pipeline's own outputs onto the five categories."""
    if scan_json.get('overall_status', 'SAFE') == 'SAFE':
        return 'NO_COLLISION', ''
    if not kur_json.get(kur_key, False):
        return 'KURAMOTO', ''
    fv = ret_json.get('final_verification', {})
    if fv.get('verified_collision_free', False):
        return 'RETRACTION', ''
    return 'UNRESOLVED', ret_json.get('unresolved_reason', 'residual pairs after retraction')


def summarise(rows, title, out_prefix):
    n = len(rows)
    print('\n' + '=' * 74)
    print(f'  {title}   ({n} trials)')
    print('=' * 74)
    counts = {c: sum(1 for r in rows if r['category'] == c) for c in CAT}
    coll = n - counts['NO_COLLISION'] - counts['IK_FAIL']
    print(f'\n  IK failures            : {counts["IK_FAIL"]:3d}')
    print(f'  Collision-free from IK : {counts["NO_COLLISION"]:3d}  ({pct(counts["NO_COLLISION"], n)})')
    print(f'  Had collision(s)       : {coll:3d}  ({pct(coll, n)})')
    if coll:
        print(f'     resolved by Kuramoto alone      : {counts["KURAMOTO"]:3d}  ({pct(counts["KURAMOTO"], coll)} of colliding)')
        print(f'     resolved by retraction+Kuramoto : {counts["RETRACTION"]:3d}  ({pct(counts["RETRACTION"], coll)} of colliding)')
        print(f'     UNRESOLVED                      : {counts["UNRESOLVED"]:3d}  ({pct(counts["UNRESOLVED"], coll)} of colliding)')
    ok = [r for r in rows if r['category'] in ('NO_COLLISION', 'KURAMOTO', 'RETRACTION')]
    if ok:
        d = [r['duration_s'] for r in ok if r['duration_s']]
        c = [r['clearance_cm'] for r in ok if r['clearance_cm'] is not None]
        print(f'\n  duration  : mean {np.mean(d):.2f}s  min {np.min(d):.2f}s  max {np.max(d):.2f}s')
        print(f'  clearance : mean {np.mean(c):.1f}cm  min {np.min(c):.1f}cm')
    for cat in ('KURAMOTO', 'RETRACTION'):
        sub = [r for r in rows if r['category'] == cat]
        if sub:
            print(f'  {cat:11s}: mean duration {np.mean([r["duration_s"] for r in sub]):.2f}s  '
                  f'mean overhead {np.mean([r["overhead"] for r in sub]) * 100:+.0f}%')
    un = [r for r in rows if r['category'] == 'UNRESOLVED']
    if un:
        print('\n  UNRESOLVED reasons:')
        for r in un: print(f'    trial {r["trial"]:3d}: {r["reason"]}')
    with open(f'{out_prefix}_results.json', 'w') as fh:
        json.dump({'n_trials': n, 'counts': counts, 'trials': rows}, fh, indent=2)
    with open(f'{out_prefix}_results.csv', 'w') as fh:
        fh.write('trial,category,duration_s,overhead,clearance_cm,colliding_pairs,reason\n')
        for r in rows:
            fh.write(f'{r["trial"]},{r["category"]},{r["duration_s"]},{r["overhead"]},'
                     f'{r["clearance_cm"]},"{r["pairs"]}","{r["reason"]}"\n')
    print(f'\n  Saved: {out_prefix}_results.json  {out_prefix}_results.csv\n')
    return counts


def pct(a, b):
    return f'{100.0 * a / b:.0f}%' if b else 'n/a'