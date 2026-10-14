#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'six'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json, time
import numpy as np
from scipy.interpolate import BSpline
from _robot8x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ,
                      ROBOT_BASES, ARM_NAMES, LINK_RADII, SAFETY_MARGIN,
                      pair_collides, witness_pair, escape_dq,
                      CLEAR_M, DEG, SEED_NCP, RESOLVE_NS, SCAN_STRIDE,
                      PREFILTER_M, _caps_surface, make_knots, basis_row,
                      eval_cp, eval_cp_n, mesh_scan, residual_of, resample,
                      limit_scale, simultaneity)
from step_124 import kuramoto_sweep

DT = 0.01
MIN_SAFE = LINK_RADII.min() * 2 + SAFETY_MARGIN
S25_BUDGET = float(__import__('os').environ.get('DUAL_ARM_TIME_BUDGET_S', '600'))
NO_PROGRESS_STOP = int(__import__('os').environ.get('DUAL_ARM_NOPROGRESS_ROUNDS', '2'))
VEL_SAFETY = 1.10
ALPHA0 = float(__import__('os').environ.get('DUAL_ARM_ESCAPE_ALPHA', '0.15'))
LS_STEPS = int(__import__('os').environ.get('DUAL_ARM_LINE_SEARCH', '5'))
MAX_REPAIR = int(__import__('os').environ.get('DUAL_ARM_ESCAPE_ITER', '25'))
REFINE_CP = int(__import__('os').environ.get('DUAL_ARM_REFINE_CP', '4'))
REFINE_MAX = int(__import__('os').environ.get('DUAL_ARM_REFINE_MAX', '28'))


def global_cp_from(td):
    sp = td.get('spline', {})
    if 'global_cp' in sp: return np.array(sp['global_cp'], float)
    cps = np.array(td['trajectory']['positions'])
    idx = np.linspace(0, len(cps) - 1, SEED_NCP).astype(int)
    return cps[idx].copy()


def refine_cp(cp, u, n_extra=None):
    if n_extra is None: n_extra = REFINE_CP
    ncp = len(cp)
    if n_extra <= 0 or ncp + n_extra > REFINE_MAX: return cp
    ncp2 = ncp + n_extra
    ns = max(200, 20 * ncp2); s = np.linspace(0.0, 1.0, ns)
    P = eval_cp_n(cp, ns)
    kn2 = make_knots(ncp2)
    B = np.asarray(BSpline.design_matrix(np.clip(s, 0, 1 - 1e-9), kn2, DEG).todense())
    cp2, *_ = np.linalg.lstsq(B, P, rcond=None)
    cp2[0] = cp[0]; cp2[-1] = cp[-1]
    return np.clip(cp2, POS_LIM[:, 0], POS_LIM[:, 1])


def apply_curve_delta(cp, u, dq):
    row = basis_row(len(cp), u).copy()
    row[0] = 0.0; row[-1] = 0.0
    denom = float(row @ row)
    if denom < 1e-12: return cp, False
    out = cp.copy()
    for c in range(len(cp)):
        if row[c] == 0.0: continue
        out[c] = np.clip(cp[c] + row[c] * dq / denom, POS_LIM[:, 0], POS_LIM[:, 1])
    return out, True


def worst_arc(cp_a, cp_b, ba, bb, ns=RESOLVE_NS):
    Pa = eval_cp_n(cp_a, ns); Pb = eval_cp_n(cp_b, ns)
    best = None; bk = -1
    for k in range(1, ns - 1):
        if _caps_surface(Pa[k], ba, Pb[k], bb) > PREFILTER_M: continue
        w = witness_pair(Pa[k], ba, Pb[k], bb)
        if best is None or w['dist'] < best:
            best = w['dist']; bk = k
    return bk, best


def escape_repair(cp, arms, pairs, bases, names, duration, deadline):
    def global_cs(cp_all):
        P = {n: eval_cp(cp_all[n], duration) for n in names}
        cs, _ = mesh_scan(P, names, bases, stride=SCAN_STRIDE)
        return cs

    work = {n: cp[n].copy() for n in names}
    best_cs = global_cs(work)
    for it in range(MAX_REPAIR):
        if time.time() > deadline or best_cs == 0: break
        improved = False
        for (a, b) in pairs:
            k, sep = worst_arc(work[a], work[b], bases[a], bases[b])
            if k < 0: continue
            u = k / (RESOLVE_NS - 1)
            if REFINE_CP > 0:
                work[a] = refine_cp(work[a], u); work[b] = refine_cp(work[b], u)
            Pa = eval_cp_n(work[a], RESOLVE_NS); Pb = eval_cp_n(work[b], RESOLVE_NS)
            alpha = ALPHA0
            for ls in range(LS_STEPS):
                dq_a, dq_b, w = escape_dq(Pa[k], bases[a], Pb[k], bases[b],
                                          alpha=alpha)
                if float(np.linalg.norm(dq_a)) < 1e-9 and \
                   float(np.linalg.norm(dq_b)) < 1e-9:
                    break
                trial = {n: work[n].copy() for n in names}
                ca, oka = apply_curve_delta(trial[a], u, dq_a)
                cb, okb = apply_curve_delta(trial[b], u, dq_b)
                if not (oka or okb): break
                trial[a] = ca; trial[b] = cb
                cs = global_cs(trial)
                if cs < best_cs:
                    work = trial; best_cs = cs; improved = True; break
                alpha *= 0.5
        if not improved: break
    return work, best_cs


def escape_static(cp_all, mover, obstacle_q, obstacle_base, names, bases,
                  duration, deadline):
    work = {n: cp_all[n].copy() for n in names}
    base_mover = bases[mover]
    def hits(cpm):
        P = eval_cp_n(cpm, RESOLVE_NS)
        return sum(1 for k in range(1, RESOLVE_NS - 1)
                   if pair_collides(P[k], base_mover, obstacle_q, obstacle_base,
                                    margin=CLEAR_M))
    best = hits(work[mover])
    for it in range(MAX_REPAIR):
        if best == 0 or time.time() > deadline: break
        P = eval_cp_n(work[mover], RESOLVE_NS)
        bk = -1; bd = None
        for k in range(1, RESOLVE_NS - 1):
            w = witness_pair(P[k], base_mover, obstacle_q, obstacle_base)
            if bd is None or w['dist'] < bd: bd = w['dist']; bk = k
        if bk < 0: break
        u = bk / (RESOLVE_NS - 1); alpha = ALPHA0; moved = False
        for ls in range(LS_STEPS):
            dq_m, _, w = escape_dq(P[bk], base_mover, obstacle_q, obstacle_base,
                                   alpha=alpha, both=False)
            if float(np.linalg.norm(dq_m)) < 1e-9: break
            cand, ok = apply_curve_delta(work[mover], u, dq_m)
            if not ok: break
            h = hits(cand)
            if h < best: work[mover] = cand; best = h; moved = True; break
            alpha *= 0.5
        if not moved: break
    return work[mover], best


def clusters_of(residual):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in residual:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    groups = {}
    for a, b in residual:
        groups.setdefault(find(a), set()).update([a, b])
    out = []
    for root, arms in groups.items():
        pr = [p for p in residual if find(p[0]) == root]
        out.append((sorted(arms), pr))
    out.sort(key=lambda t: -len(t[0]))
    return out


def stage_C(cp, names, bases, starts, targets, residual, duration, deadline):
    def parked_hits(mover, owner, pose, ns=3 * RESOLVE_NS):
        P = eval_cp_n(cp[mover], ns)
        return sum(1 for k in range(1, ns - 1)
                   if pair_collides(P[k], bases[mover], pose, bases[owner],
                                    margin=CLEAR_M))
    cons = []
    for (na, nb) in residual:
        st = {n: starts[n] for n in (na, nb)}; tg = {n: targets[n] for n in (na, nb)}
        def attempt(first, second):
            for _ in range(2):
                h1 = parked_hits(first, second, st[second])
                h2 = parked_hits(second, first, tg[first])
                if h1 == 0 and h2 == 0: break
                if h1 > 0:
                    cp[first], _ = escape_static(cp, first, st[second],
                                                 bases[second], names, bases,
                                                 duration, deadline)
                if h2 > 0:
                    cp[second], _ = escape_static(cp, second, tg[first],
                                                  bases[first], names, bases,
                                                  duration, deadline)
            return (parked_hits(first, second, st[second]),
                    parked_hits(second, first, tg[first]))
        v1 = parked_hits(na, nb, st[nb]) + parked_hits(nb, na, tg[na])
        v2 = parked_hits(nb, na, st[na]) + parked_hits(na, nb, tg[nb])
        o1 = (na, nb) if v1 <= v2 else (nb, na)
        snap = {na: cp[na].copy(), nb: cp[nb].copy()}
        h1, h2 = attempt(*o1); first, second = o1
        if h1 + h2 > 0:
            keep = (h1 + h2, {na: cp[na].copy(), nb: cp[nb].copy()}, o1)
            cp[na] = snap[na].copy(); cp[nb] = snap[nb].copy()
            g1, g2 = attempt(o1[1], o1[0])
            if g1 + g2 < keep[0]: first, second = o1[1], o1[0]; h1, h2 = g1, g2
            else:
                cp[na], cp[nb] = keep[1][na], keep[1][nb]
                first, second = keep[2]; h1, h2 = attempt(*keep[2])
        cons.append([first, second])
        print(f'  [C] {na}<->{nb}: order {first} first  parked_hits={h1}+{h2}')
    return cons


def topo_sequential(base_P, names, bases, cons0, duration, res_pairs):
    combos = [cons0]
    for m in range(len(cons0)):
        for bc in list(combos):
            fl = [list(c) for c in bc]
            fl[m] = [fl[m][1], fl[m][0]]
            if fl not in combos and len(combos) < 8: combos.append(fl)
    half = max(2, int(round(duration * RATE_HZ)))
    for cons in combos:
        depth = {n: 0 for n in names}
        for _ in range(len(names)):
            for a, b in cons:
                if depth[b] < depth[a] + 1: depth[b] = depth[a] + 1
        K = max(depth.values()) + 1 if depth else 1
        seq = {}
        for n in names:
            p = base_P[n]; blocks = []
            for ph in range(K):
                if ph < depth[n]: blocks.append(np.repeat(p[0][None, :], half, axis=0))
                elif ph == depth[n]: blocks.append(p)
                else: blocks.append(np.repeat(p[-1][None, :], half, axis=0))
            seq[n] = np.vstack(blocks)
        cs_r, _ = mesh_scan(seq, names, bases, stride=SCAN_STRIDE, pairs=res_pairs)
        if cs_r > 0: continue
        cs, _ = mesh_scan(seq, names, bases, stride=SCAN_STRIDE)
        if cs == 0: return seq, K * duration, K
    return None, None, None


def _write(names, chosen, kdur, base_dur, meta, label, resolved, method):
    out = {'duration': float(kdur), 'arm_names': names,
           'final_duration_s': round(float(kdur), 4),
           'duration_overhead': round(float(kdur) / base_dur - 1, 4),
           'synchronisation_report': {'collision_free': resolved, 'method': method},
           'final_verification': {'verified_collision_free': resolved}}
    for n in names:
        pos = np.asarray(chosen[n]); dt = kdur / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[n] = {'robot_name': n, 'metadata': {**meta[n], 'duration': float(kdur)},
                  'trajectory': {'time': np.linspace(0., kdur, len(pos)).tolist(),
                                 'positions': pos.tolist(), 'velocities': vel.tolist(),
                                 'accelerations': acc.tolist(),
                                 'arc_fracs': np.linspace(0., 1., len(pos)).tolist()}}
    with open('s125_synchronized.json', 'w') as fh: json.dump(out, fh)
    print(f'  Saved: s125_synchronized.json (6-arm)')
def main():
    s24 = json.load(open('s124_resolved.json')); names = s24['arm_names']
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in names}
    if not s24.get('needs_retraction', False):
        out = dict(s24)
        out['final_verification'] = {'verified_collision_free': True,
                                     'reason': 'kuramoto_alone_in_step_124'}
        json.dump(out, open('s125_synchronized.json', 'w'))
        print('  STEP 125: step 124 resolved all by timing -> passthrough.')
        return
    s22 = json.load(open('s122_trajectories.json'))
    duration = float(s22['duration'])
    cp = {n: global_cp_from(s22[n]) for n in names}
    cp0 = {n: cp[n].copy() for n in names}
    starts = {n: np.array(s22[n]['metadata']['start_joints'], float) for n in names}
    targets = {n: np.array(s22[n]['metadata']['end_joints'], float) for n in names}
    meta = {n: s22[n]['metadata'] for n in names}
    residual0 = [tuple(p) for p in s24.get('residual_pairs', [])]
    print(f'  STEP 125: timing-insufficient pairs {residual0}')
    deadline = time.time() + S25_BUDGET; t0 = time.time()
    best = None; best_cs = 10**9; best_dur = duration; cons = []
    rnd = 0; stale = 0; last_cs = None
    while time.time() < deadline:
        rnd += 1
        paths = {n: eval_cp(cp[n], duration) for n in names}
        residual = residual_of(paths, names, bases)
        if not residual:
            cs, _ = mesh_scan(paths, names, bases)
            if cs == 0: best, best_cs, best_dur = paths, 0, duration; break
        for arms, prs in clusters_of(residual):
            cp, gcs = escape_repair(cp, arms, prs, bases, names, duration, deadline)
            tag = 'JOINTLY' if len(arms) > 2 else ''
            print(f'  round {rnd}: cluster {arms} ({len(prs)} pairs) escaped '
                  f'{tag} -> global_cs~{gcs} ({time.time()-t0:.0f}s)')
        paths = {n: eval_cp(cp[n], duration) for n in names}
        residual = residual_of(paths, names, bases)
        print(f'  round {rnd}: after escape residual={residual}')
        chosen, kdur, cfg = kuramoto_sweep(paths, duration, names, bases,
                                           residual or None, deadline)
        if chosen is not None:
            best, best_cs, best_dur = chosen, 0, kdur
            print(f'  KURAMOTO re-iteration SUCCESS cfg={cfg}'); break
        cs, _ = mesh_scan(paths, names, bases)
        if cs == 0: best, best_cs, best_dur = paths, 0, duration; break
        if cs < best_cs: best, best_cs, best_dur = paths, cs, duration
        if rnd >= 2 and residual and time.time() < deadline:
            cons = stage_C(cp, names, bases, starts, targets, residual,
                           duration, deadline)
            paths = {n: eval_cp(cp[n], duration) for n in names}
            residual = residual_of(paths, names, bases)
            chosen, kdur, cfg = kuramoto_sweep(paths, duration, names, bases,
                                               residual or None, deadline)
            if chosen is not None:
                best, best_cs, best_dur = chosen, 0, kdur
                print(f'  KURAMOTO after parked-pose clearance SUCCESS cfg={cfg}'); break
            base_P = {n: resample(paths[n], duration) for n in names}
            seq, sdur, K = topo_sequential(base_P, names, bases, cons, duration,
                                           residual or None)
            if seq is not None:
                best, best_cs, best_dur = seq, 0, sdur
                print(f'  TOPOLOGICAL SEQUENTIAL SUCCESS (K={K} phases)'); break
        if last_cs is not None and best_cs >= last_cs:
            stale += 1
            if stale >= NO_PROGRESS_STOP:
                print(f'  no progress in {stale} rounds -> stopping early '
                      f'(structurally unresolvable; {best_cs} coll steps)'); break
        else: stale = 0
        last_cs = best_cs
        if rnd >= 8: break
    dev = {}
    for n in names:
        opt = eval_cp(cp0[n], duration); cur = eval_cp(cp[n], duration)
        K = min(len(opt), len(cur)); d = cur[:K] - opt[:K]
        nrm = np.linalg.norm(d, axis=1)
        if float(nrm.max()) > 1e-6:
            dev[n] = (round(float(nrm.mean()), 4), round(float(nrm.max()), 4))
    print(f'  deviation from optimal (mean, peak) rad: {dev if dev else "none"}')
    resolved = best_cs == 0 and best is not None
    if best is None:
        best = {n: eval_cp(cp[n], duration) for n in names}; best_dur = duration
    sc = limit_scale(best, names, best_dur)
    if sc > 1.0:
        best_dur = best_dur * sc * VEL_SAFETY
        best = {n: resample(best[n], best_dur) for n in names}
    if resolved:
        print(f'  SIMULTANEITY: {100*simultaneity(best, names):.0f}%')
    print(f'  total resolver time: {time.time()-t0:.0f}s (budget {S25_BUDGET:.0f}s)')
    method = 'kuramoto_plus_witness_escape' if resolved else 'unresolved'
    label = ('Kuramoto + perpendicular escape SUCCESS' if resolved
             else f'UNRESOLVED ({best_cs} coll steps)')
    _write(names, best, best_dur, duration, meta, label, resolved, method)


if __name__ == '__main__':
    main()