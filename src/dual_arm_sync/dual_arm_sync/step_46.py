#!/usr/bin/env python3

import io, json, os, sys, tempfile, time, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robot2x as R
from _bench_common import quiet, sample_target, scan_all, classify, summarise, CAT

TRIALS   = int(os.environ.get('DUAL_ARM_TRIALS', '50'))
SEED     = int(os.environ.get('DUAL_ARM_SEED', '0'))
DURATION = float(os.environ.get('DUAL_ARM_DURATION', '10.0'))
CLEAR_M  = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
HERE     = os.path.dirname(os.path.abspath(__file__))

START = {'dsr01': np.array([-0.06261719, -0.02364304, 1.87302947, -0.02662147, 1.49003406, 0.0]),
         'dsr02': np.array([0.06261735, 0.02364306, -1.87303491, 0.02662157, -1.49003859, 0.0])}


def one_trial(t, rng, mods):
    step_40, step_41, step_42, step_43, step_44 = mods
    names = list(R.ARM_NAMES)
    bases = {n: np.array(R.ROBOT_BASES[n]) for n in names}
    tgt = {n: sample_target(rng, R, bases[n], START[n]) for n in names}
    row = {'trial': t, 'category': 'IK_FAIL', 'duration_s': None, 'overhead': None,
           'clearance_cm': None, 'pairs': '', 'reason': '',
           'targets': {n: np.round(tgt[n], 3).tolist() for n in names}}

    arms = {n: {'base': bases[n], 'start_q': START[n], 'target_local': tgt[n] - bases[n],
                'target_world': tgt[n].tolist(), 'target_rot': None,
                'constrain_orient': False} for n in names}
    ok, _ = quiet(lambda: None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            best = step_40.greedy_pairwise_selection(arms)
    except Exception as e:
        row['reason'] = f'IK exception: {type(e).__name__}'; return row
    if best is None:
        row['reason'] = 'no collision-free endpoint assignment (CSP failed)'; return row

    ik = {'duration': DURATION, 'arm_names': names}
    for n in names:
        ik[n] = {'base': bases[n].tolist(),
                 'start_joints': START[n].tolist(),
                 'target_joints': np.asarray(best[n]).tolist()}
    with open('s40_ik.json', 'w') as fh: json.dump(ik, fh)

    for mod, tag in ((step_41, 's41'), (step_42, 's42'), (step_43, 's43'), (step_44, 's44')):
        good, out = quiet(mod.main)
        if not good:
            row['category'] = 'UNRESOLVED'; row['reason'] = f'{tag} aborted'; return row

    sfile = 's42_collision_map.json' if os.path.exists('s42_collision_map.json') else 's42_collision.json'
    scan = json.load(open(sfile)) if os.path.exists(sfile) else {}
    if 'overall_status' not in scan:
        pl = scan.get('pairs', [])
        scan['overall_status'] = 'COLLISION' if any(p.get('status') == 'COLLISION' for p in pl) else 'SAFE'
    kur = json.load(open('s43_kuramoto.json'))
    if not os.path.exists('s44_synchronized.json'):
        row['category'] = 'UNRESOLVED'
        row['reason'] = 'step_44 could not clear (deep swap / target overlap)'
        row['pairs'] = ';'.join(p['pair'] for p in scan.get('pairs', []) if p.get('status') == 'COLLISION')
        return row
    ret = json.load(open('s44_synchronized.json'))
    cat, reason = classify(scan, kur, ret)
    row['category'] = cat; row['reason'] = reason
    row['pairs'] = ';'.join(p['pair'] for p in scan.get('pairs', []) if p.get('status') == 'COLLISION')

    src = ret if ret else kur
    if cat in ('NO_COLLISION', 'KURAMOTO', 'RETRACTION') and src:
        P = {n: np.array(src[n]['trajectory']['positions']) for n in names}
        bad, worst = scan_all(R, names, bases, P, CLEAR_M)
        row['duration_s'] = round(float(src.get('final_duration_s', DURATION)), 3)
        row['overhead'] = round(row['duration_s'] / DURATION - 1.0, 3)
        row['clearance_cm'] = round(worst * 100, 1)
        if bad:                                
            row['category'] = 'UNRESOLVED'; row['reason'] = f'post-check collision {bad}'
    return row


def main():
    print('\n' + '=' * 74)
    print(f'  STEP 46  --  Randomised benchmark, DUAL ARM (step_40..45), {TRIALS} trials')
    print('=' * 74)
    import step_40, step_41, step_42, step_43, step_44
    mods = (step_40, step_41, step_42, step_43, step_44)
    rng = np.random.default_rng(SEED)
    rows = []; t0 = time.time(); cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='s46_')
    try:
        os.chdir(tmp)
        for t in range(1, TRIALS + 1):
            r = one_trial(t, rng, mods)
            rows.append(r)
            print(f'  [{t:3d}/{TRIALS}] {r["category"]:12s} '
                  f'dur={r["duration_s"] if r["duration_s"] else "-":>6} '
                  f'clr={r["clearance_cm"] if r["clearance_cm"] is not None else "-":>5} '
                  f'{r["reason"][:38]}', flush=True)
            for f in ('s40_ik.json', 's41_trajectories.json', 's42_collision.json',
                      's42_collision_map.json', 's43_kuramoto.json', 's44_synchronized.json'):
                if os.path.exists(f): os.remove(f)
    finally:
        os.chdir(cwd)
    print(f'\n  wall time {time.time() - t0:.1f}s')
    summarise(rows, 'DUAL-ARM PIPELINE (step_40..45)', os.path.join(cwd, 's46'))


if __name__ == '__main__':
    main()