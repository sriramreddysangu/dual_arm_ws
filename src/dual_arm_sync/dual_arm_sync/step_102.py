#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'dual'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json
import numpy as np
from scipy.interpolate import BSpline
from _robot8x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                      ARM_NAMES, fk_world)

DEG = 3
SEED_NCP = 12


def make_knots(ncp):
    ni = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def seed_cp(start_q, end_q, ncp=SEED_NCP):
    s = np.linspace(0., 1., ncp)
    cp = start_q[None, :] + s[:, None] * (end_q - start_q)[None, :]
    cp[1] = cp[0]; cp[-2] = cp[-1]      
    return np.clip(cp, POS_LIM[:, 0], POS_LIM[:, 1])


def eval_cp(cp, duration):
    kn = make_knots(len(cp)); n = max(2, int(round(duration * RATE_HZ)))
    s = np.linspace(0., 1., n)
    pos = np.zeros((n, NDOF)); vel = np.zeros((n, NDOF)); acc = np.zeros((n, NDOF))
    for j in range(NDOF):
        spl = BSpline(kn, cp[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s)
        vel[:, j] = spl.derivative(1)(s) / duration
        acc[:, j] = spl.derivative(2)(s) / duration**2
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1]), vel, acc


def scale_duration(cp, duration):
    pos, vel, acc = eval_cp(cp, duration); sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc = eval_cp(cp, duration)
    return pos, vel, acc, duration


def resample(pos, nout):
    sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
    return np.clip(np.vstack([np.interp(sout, sin, pos[:, j])
                              for j in range(NDOF)]).T,
                   POS_LIM[:, 0], POS_LIM[:, 1])

def main():
    with open('s101_ik.json') as fh: ik = json.load(fh)
    arm_names = ik.get('arm_names', ARM_NAMES); requested = float(ik.get('duration', 10.0))
    out = {'duration': requested, 'arm_names': arm_names}; max_dur = 0.
    for name in arm_names:
        d = ik[name]
        s = np.array(d['start_joints'], float); e = np.array(d['target_joints'], float)
        base = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
        cp = seed_cp(s, e)
        pos, vel, acc, dur = scale_duration(cp, requested); n = len(pos)
        out[name] = {'robot_name': name,
                     'metadata': {'start_joints': s.tolist(), 'end_joints': e.tolist(),
                                  'target_ee_world': fk_world(e, base).tolist(),
                                  'duration': float(dur), 'n_samples': int(n),
                                  'degree': DEG},
                     'spline': {'degree': DEG, 'n_cp': int(len(cp)),
                                'global_cp': cp.tolist()},
                     'trajectory': {'time': np.linspace(0., dur, n).tolist(),
                                    'positions': pos.tolist(),
                                    'velocities': vel.tolist(),
                                    'accelerations': acc.tolist(),
                                    'arc_fracs': np.linspace(0., 1., n).tolist()}}
        max_dur = max(max_dur, dur)
    for name in arm_names:                     
        if out[name]['metadata']['duration'] < max_dur - 1e-6:
            pos = np.array(out[name]['trajectory']['positions'])
            p2 = resample(pos, max(2, int(round(max_dur * RATE_HZ))))
            dt = max_dur / max(len(p2) - 1, 1)
            v2 = np.gradient(p2, dt, axis=0); a2 = np.gradient(v2, dt, axis=0)
            out[name]['trajectory'].update({
                'time': np.linspace(0, max_dur, len(p2)).tolist(),
                'positions': p2.tolist(), 'velocities': v2.tolist(),
                'accelerations': a2.tolist(),
                'arc_fracs': np.linspace(0, 1, len(p2)).tolist()})
            out[name]['metadata']['duration'] = max_dur
    out['duration'] = max_dur
    with open('s102_trajectories.json', 'w') as fh: json.dump(out, fh)
    print(f'  STEP 102: {len(arm_names)} splines, duration {max_dur:.2f}s')


if __name__ == '__main__':
    main()