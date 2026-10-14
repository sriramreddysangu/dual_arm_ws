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
from _robot8x import (ROBOT_BASES, ARM_NAMES, mesh_scan, residual_of,
                      resample, limit_scale, simultaneity)
import step_104 as S104

OUT_PATH = 's254_resolved.json'


def _write_resolved(names, chosen, kdur, base_dur, meta, splines,
                    needs_retraction, residual_pairs, method, mind, brake_trace):
    """step_104._write does not take a brake_trace, and the 6-arm one does.
    Writing the record here keeps both cores usable unchanged and keeps the
    schema identical to what step_255 expects to read back."""
    out = {'duration': float(kdur), 'arm_names': names,
           'final_duration_s': round(float(kdur), 4),
           'duration_overhead': round(float(kdur) / base_dur - 1, 4),
           'needs_retraction': bool(needs_retraction),
           'residual_pairs': residual_pairs,
           'synchronisation_report': {
               'collision_free': not needs_retraction,
               'min_dist_m': round(float(mind), 4) if mind is not None else None,
               'method': method}}
    if brake_trace:
        out['brake_trace'] = brake_trace
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
    with open(OUT_PATH, 'w') as fh:
        json.dump(out, fh)
    print('  Saved: {}'.format(OUT_PATH))


def _predictions(cmap):
    out = {}
    for p in cmap.get('pairs', []):
        tf = p.get('timing_feasibility')
        if tf:
            key = tuple(sorted((p['arm_i'], p['arm_j'])))
            out[key] = tf
    return out


def _report_prediction(pred, coll_pairs, resolved_by_timing):
    if not pred:
        return
    print('\n  PREDICTED (step_253)  vs  ACTUAL (step_254):')
    for (a, b) in coll_pairs:
        tf = pred.get(tuple(sorted((a, b))))
        if tf is None:
            continue
        verdict = tf['verdict']
        if resolved_by_timing:
            actual = 'timing alone cleared the cell'
            agree = (verdict == 'KURAMOTO_ALONE')
        else:
            actual = 'handed to step_255'
            agree = True if verdict == 'NEEDS_DEFORMATION' else None
        flag = ('  [as predicted]' if agree is True
                else ('  [MISMATCH -- check the grid resolution]'
                      if agree is False else
                      '  [pair was resolvable alone; the CELL was not]'))
        print('    {}<->{}: predicted {:<17} -> {}{}'.format(
            a, b, verdict, actual, flag))

def main():
    with open('s252_trajectories.json') as fh:
        t = json.load(fh)
    with open('s253_collision_map.json') as fh:
        cmap = json.load(fh)
    names = t.get('arm_names', ARM_NAMES)
    duration = float(t['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in names}
    paths = {n: np.array(t[n]['trajectory']['positions'], float) for n in names}
    meta = {n: t[n]['metadata'] for n in names}
    splines = {n: t[n].get('spline', {}) for n in names}
    coll_pairs = [(p['arm_i'], p['arm_j']) for p in cmap.get('pairs', [])
                  if p.get('status') == 'COLLISION']
    pred = _predictions(cmap)

    base_P = {n: resample(paths[n], duration) for n in names}
    if not coll_pairs:
        _write_resolved(names, base_P, duration, duration, meta, splines,
                        False, [], 'none_needed', None, [])
        print('  STEP 254: seed is already collision-free -- nothing to resolve.')
        print('  Next: step_255 (will pass through)')
        return

    print('  STEP 254: {} colliding pair(s): {}'.format(
        len(coll_pairs), ', '.join('{}<->{}'.format(a, b) for a, b in coll_pairs)))
    deadline = time.time() + S104.S24_BUDGET
    t0 = time.time()
    chosen, kdur, cfg, nm, mind = S104.kuramoto_sweep(
        paths, duration, names, bases, coll_pairs, deadline, want_nm=True)

    if chosen is not None:
        sc = limit_scale(chosen, names, kdur, compress=True)
        if sc > 1e-6 and (sc * S104.VEL_SAFETY < 0.98 or sc > 1.0):
            cand_dur = max(kdur * sc * S104.VEL_SAFETY, 0.5)
            cand = {n: resample(chosen[n], cand_dur) for n in names}
            cs2, mind2 = mesh_scan(cand, names, bases)
            if cs2 == 0:
                chosen, kdur, mind = cand, cand_dur, mind2
            elif sc > 1.0:
                kdur = kdur * sc * S104.VEL_SAFETY
                chosen = {n: resample(chosen[n], kdur) for n in names}
        print('  STEP 254 KURAMOTO SUCCESS (timing alone) cfg={} min={:.1f}cm '
              'dur {:.1f}s -> {:.2f}s  ({:.0f}s)'.format(
                  cfg, mind * 100, duration, kdur, time.time() - t0))
        print('  SIMULTANEITY: {:.0f}% (lag_cap={}; 1.0 = take-turns)'.format(
            100 * simultaneity(chosen, names), cfg[2]))
        _report_prediction(pred, coll_pairs, True)

        brake_trace = []
        trace_fn = getattr(S104, 'kuramoto_trace', None)
        label_fn = getattr(S104, '_held_from_label', None)
        if trace_fn is not None and label_fn is not None:
            _, _, brake_trace = trace_fn(paths, duration, names, bases,
                                         cfg[0], cfg[1], cfg[2],
                                         label_fn(cfg[3]), conflict=coll_pairs)
            if brake_trace:
                arms_braked = sorted({e['arm'] for e in brake_trace})
                print('  BRAKE TRACE: {} event(s) across {} -- first at '
                      't={:.2f}s, last at t={:.2f}s'.format(
                          len(brake_trace), ', '.join(arms_braked),
                          brake_trace[0]['t_s'], brake_trace[-1]['t_s']))
        _write_resolved(names, chosen, kdur, duration, meta, splines,
                        False, [], 'kuramoto_phase_lag', mind, brake_trace)
        print('  Next: step_255 (will pass through) then step_256')
        return

    residual = residual_of(base_P, names, bases)
    print('  STEP 254: timing cannot separate {} -- deferring to step_255.'
          .format(residual))
    _report_prediction(pred, coll_pairs, False)
    _write_resolved(names, base_P, duration, duration, meta, splines,
                    True, [list(p) for p in residual],
                    'kuramoto_insufficient', None, [])
    print('  Next: step_255 (deformation + re-synchronisation)')


if __name__ == '__main__':
    main()