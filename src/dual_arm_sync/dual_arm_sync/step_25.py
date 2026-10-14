#!/usr/bin/env python3

import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                      ARM_NAMES, pair_min_dist, pair_collides,
                      _arm_caps, caps_min_dist, LINK_RADII, SAFETY_MARGIN)

CLEAR_M  = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
BUDGET_S = float(os.environ.get('DUAL_ARM_TIME_BUDGET_S', '600'))
NO_PROGRESS_STOP = int(os.environ.get('DUAL_ARM_NOPROGRESS_ROUNDS', '2'))
DT       = 0.01
MIN_SAFE = LINK_RADII.min() * 2 + SAFETY_MARGIN
MAX_LAG = float(os.environ.get('DUAL_ARM_MAX_LAG', '1.0'))
LAG_SWEEP = [(10.0, 0.06, 0.45),
             (12.0, 0.08, 0.28),
             (10.0, 0.10, 0.70),
             (12.0, 0.10, 1.00),
             (16.0, 0.15, 1.00)]
VEL_SAFETY = 1.10
DEG = 3
MAX_ITER = int(os.environ.get('DUAL_ARM_RETRACT_ITER', '4'))
SPAN_PAD = 0.10; RESOLVE_NS = 220
RESOLVE_BUFFER = 0.03; STALL_ITERS = 8
PULL_GAIN = 0.5; GAIN_MAX = 1.0; GAIN_STEP = 0.15

def make_knots(ncp):
    ni = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def greville(kn):
    n = len(kn) - DEG - 1
    return np.array([float(np.mean(kn[i + 1:i + DEG + 1])) for i in range(n)])


def eval_cp_n(cp, n):
    kn = make_knots(len(cp)); s = np.linspace(0., 1., n); P = np.zeros((n, NDOF))
    for j in range(NDOF):
        P[:, j] = BSpline(kn, cp[:, j], DEG, extrapolate=True)(s)
    return np.clip(P, POS_LIM[:, 0], POS_LIM[:, 1])


def eval_cp(cp, duration):
    return eval_cp_n(cp, max(2, int(round(duration * RATE_HZ))))


def global_cp_from(td):
    sp = td.get('spline', {})
    if 'global_cp' in sp: return np.array(sp['global_cp'], float)
    cps = np.array(td['trajectory']['positions'])
    idx = np.linspace(0, len(cps) - 1, 12).astype(int)
    return cps[idx].copy()


def interp(path, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(path) - 1
    if n <= 0: return path[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return path[idx] + a * (path[min(idx + 1, n)] - path[idx])


def resample(pos, duration):
    nout = max(2, int(round(duration * RATE_HZ)))
    sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
    return np.clip(np.vstack([np.interp(sout, sin, pos[:, j]) for j in range(NDOF)]).T,
                   POS_LIM[:, 0], POS_LIM[:, 1])

def mesh_scan(P, names, bases, stride=2, pairs=None):
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


def residual_of(P, names, bases):
    res = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c, _ = mesh_scan(P, names, bases, stride=2,
                             pairs=[(names[i], names[j])])
            if c > 0:
                res.append((names[i], names[j]))
    return res


def limit_scale(P, names, duration):
    s = 1.0
    for n in names:
        pos = P[n]; dt = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        for j in range(NDOF):
            vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
            if vp > VEL_LIM[j]: s = max(s, vp / VEL_LIM[j])
            if ap > ACC_LIM[j]: s = max(s, float(np.sqrt(ap / ACC_LIM[j])))
    return s

def lead_combos(coll_pairs, cap=None):
    if cap is None:
        cap = int(os.environ.get('DUAL_ARM_LEAD_COMBOS', '16'))
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


def simultaneity(P, names):
    K = min(len(P[n]) for n in names)
    if K < 3:
        return 1.0
    mov = np.zeros((K - 1, len(names)), bool)
    for i, n in enumerate(names):
        mov[:, i] = np.max(np.abs(np.diff(np.asarray(P[n])[:K], axis=0)), axis=1) > 1e-5
    nm = mov.sum(axis=1); act = nm >= 1
    return float((nm[act] > 1).mean()) if act.any() else 1.0


def kuramoto(paths, duration, names, bases, k_lag, band, lag_cap,
             held_set=frozenset()):
    d_safe = MIN_SAFE + band
    N = len(names); phi = np.zeros(N); omega = np.full(N, 1.0 / duration)
    hold_steps = int(0.05 * duration / DT) if held_set else 0
    held = np.array([n in held_set for n in names], bool)
    traj = {n: [] for n in names}
    nsteps = int((2.0 + 2.0 * min(lag_cap, 1.0)) * duration / DT)
    steps_done = 0
    for step in range(nsteps):
        q = {n: interp(paths[n], phi[i]) for i, n in enumerate(names)}
        for n in names: traj[n].append(q[n].copy())
        steps_done = step + 1
        caps = {n: _arm_caps(q[n], bases[n]) for n in names}
        pd = {(a, b): caps_min_dist(caps[names[a]], caps[names[b]])
              for a in range(N) for b in range(a + 1, N)}
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
    return {n: np.array(traj[n]) for n in names}, real_dur


def kuramoto_sweep(paths, duration, names, bases, res_pairs, deadline):
    combos = lead_combos(res_pairs or [])
    for (kl, bd, cap) in sorted([c for c in LAG_SWEEP if c[2] <= MAX_LAG + 1e-9],
                                key=lambda c: c[2]):
        for held in combos:
            if time.time() > deadline:
                return None, None, None
            sync, real_dur = kuramoto(paths, duration, names, bases,
                                      kl, bd, cap, held)
            sr = {n: resample(sync[n], real_dur) for n in names}
            cs_r, _ = mesh_scan(sr, names, bases, stride=4, pairs=res_pairs)
            if cs_r > 0:
                continue
            cs, _ = mesh_scan(sr, names, bases, stride=2)
            if cs == 0:
                order = 'free' if not held else 'hold=' + ','.join(sorted(held))
                return sr, real_dur, (kl, bd, cap, order)
    return None, None, None

def win_weight(g, lo, hi):
    span = max(hi - lo, 1e-9); f = (g - lo) / span; ramp = 0.30
    up = min(1.0, f / ramp)
    if hi >= 0.97:                       
        return max(0.0, up)
    down = min(1.0, (1.0 - f) / ramp)
    return max(0.0, min(up, down))


def scan_pair_ns(pi, pj, bi, bj, margin):
    K = min(len(pi), len(pj)); ks = []
    for k in range(K):
        if pair_collides(pi[k], bi, pj[k], bj, margin=margin):
            ks.append(k)
    return ks


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


def resolve_cluster(cp, arms, pairs, bases):
    cp = {n: cp[n].copy() for n in arms}
    grev = {n: greville(make_knots(len(cp[n]))) for n in arms}
    gain = PULL_GAIN; best = 10 ** 9; stall = 0
    for it in range(MAX_ITER):
        P = {n: eval_cp_n(cp[n], RESOLVE_NS) for n in arms}
        wins = {n: [] for n in arms}; total = 0
        for (a, b) in pairs:
            ks = scan_pair_ns(P[a], P[b], bases[a], bases[b],
                              CLEAR_M + RESOLVE_BUFFER)
            if ks:
                total += len(ks)
                wins[a].append((ks[0], ks[-1])); wins[b].append((ks[0], ks[-1]))
        if total == 0:
            break
        if total < best:
            best = total; stall = 0
        else:
            stall += 1
            if stall >= STALL_ITERS:
                if gain >= GAIN_MAX:
                    break
                gain = min(GAIN_MAX, gain + GAIN_STEP); stall = 0
        arc = np.linspace(0., 1., RESOLVE_NS); moved = False
        for n in arms:
            if not wins[n]:
                continue
            lo = max(arc[min(w[0] for w in wins[n])] - SPAN_PAD, 0.0)
            hi = min(arc[max(w[1] for w in wins[n])] + SPAN_PAD, 1.0)
            for c in range(1, len(cp[n]) - 1):
                g = grev[n][c]
                if lo <= g <= hi:
                    w = win_weight(g, lo, hi)
                    if w <= 1e-6:
                        continue
                    rt = cp[n][c].copy(); rt[1:] = 0.0
                    new = np.clip(cp[n][c] + gain * w * (rt - cp[n][c]),
                                  POS_LIM[:, 0], POS_LIM[:, 1])
                    if float(np.max(np.abs(new - cp[n][c]))) > 1e-6:
                        moved = True
                    cp[n][c] = new
        if not moved:
            break
    return cp


def resolve_pair(cp_i, cp_j, bi, bj):
    cp_i = cp_i.copy(); cp_j = cp_j.copy()
    gr_i = greville(make_knots(len(cp_i))); gr_j = greville(make_knots(len(cp_j)))
    gain = PULL_GAIN; best = 10 ** 9; stall = 0
    for it in range(MAX_ITER):
        pi = eval_cp_n(cp_i, RESOLVE_NS); pj = eval_cp_n(cp_j, RESOLVE_NS)
        ks = scan_pair_ns(pi, pj, bi, bj, CLEAR_M + RESOLVE_BUFFER)
        if not ks:
            break
        if len(ks) < best:
            best = len(ks); stall = 0
        else:
            stall += 1
            if stall >= STALL_ITERS:
                if gain >= GAIN_MAX:
                    break
                gain = min(GAIN_MAX, gain + GAIN_STEP); stall = 0
        arc = np.linspace(0., 1., RESOLVE_NS)
        lo = max(arc[ks[0]] - SPAN_PAD, 0.0); hi = min(arc[ks[-1]] + SPAN_PAD, 1.0)
        for cp, grev in ((cp_i, gr_i), (cp_j, gr_j)):
            for c in range(1, len(cp) - 1):
                g = grev[c]
                if lo <= g <= hi:
                    w = win_weight(g, lo, hi)
                    rt = cp[c].copy(); rt[1:] = 0.0
                    cp[c] = np.clip(cp[c] + gain * w * (rt - cp[c]),
                                    POS_LIM[:, 0], POS_LIM[:, 1])
    return cp_i, cp_j

def column_cp(s, t):
    mid = np.zeros(NDOF); mid[0] = 0.5 * (s[0] + t[0])
    a = np.linspace(0., 1., 6)[:, None]
    up = s[None, :] + a * (mid - s)[None, :]
    dn = mid[None, :] + a * (t - mid)[None, :]
    cp = np.vstack([up, np.tile(mid, (2, 1)), dn[1:]])
    cp = np.vstack([cp[0], cp, cp[-1]])
    return np.clip(cp, POS_LIM[:, 0], POS_LIM[:, 1])


def try_reroute(cp, pair, names, bases, starts, targets):
    na, nb = pair
    def pair_cs():
        pa = eval_cp_n(cp[na], RESOLVE_NS); pb = eval_cp_n(cp[nb], RESOLVE_NS)
        return len(scan_pair_ns(pa, pb, bases[na], bases[nb], CLEAR_M))
    cs = pair_cs()
    for n in (na, nb):
        saved = cp[n].copy()
        cp[n] = column_cp(starts[n], targets[n])
        cs2 = pair_cs()
        if cs2 < cs:
            print(f'  [B] rerouted {n} through the column: {na}<->{nb} '
                  f'residual {cs}->{cs2}')
            cs = cs2
        else:
            cp[n] = saved
        if cs == 0:
            break
    return cs


def retract_static(cp0, base, obstacle_qs, obstacle_base, iters=120, gain=0.5):
    cp = cp0.copy()
    grev = greville(make_knots(len(cp)))
    END_GUARD = int(0.05 * RESOLVE_NS)
    for _ in range(iters):
        P = eval_cp_n(cp, RESOLVE_NS)
        ks = []
        for k in range(1, RESOLVE_NS - 1):
            m = (CLEAR_M if (k <= END_GUARD or k >= RESOLVE_NS - 1 - END_GUARD)
                 else CLEAR_M + RESOLVE_BUFFER)
            if any(pair_collides(P[k], base, oq, obstacle_base, margin=m)
                   for oq in obstacle_qs):
                ks.append(k)
        if not ks:
            return cp
        arc = np.linspace(0., 1., RESOLVE_NS)
        lo = max(arc[ks[0]] - SPAN_PAD, 0.0); hi = min(arc[ks[-1]] + SPAN_PAD, 1.0)
        for c in range(1, len(cp) - 1):
            g = grev[c]
            if lo <= g <= hi:
                w = win_weight(g, lo, hi)
                rt = cp[c].copy(); rt[1:] = 0.0
                cp[c] = np.clip(cp[c] + gain * w * (rt - cp[c]),
                                POS_LIM[:, 0], POS_LIM[:, 1])
    EPS = 1e-3; ALPHA = 0.06
    for _ in range(80):
        P = eval_cp_n(cp, RESOLVE_NS)
        worst_k = -1; worst_d = CLEAR_M
        for k in range(1, RESOLVE_NS - 1):
            d = min(pair_min_dist(P[k], base, oq, obstacle_base)
                    for oq in obstacle_qs)
            if d < worst_d:
                worst_d = d; worst_k = k
        if worst_k < 0:
            return cp
        q0 = P[worst_k]
        d0 = min(pair_min_dist(q0, base, oq, obstacle_base) for oq in obstacle_qs)
        grad = np.zeros(NDOF)
        for j in range(NDOF):
            qe = q0.copy(); qe[j] += EPS
            de = min(pair_min_dist(qe, base, oq, obstacle_base)
                     for oq in obstacle_qs)
            grad[j] = (de - d0) / EPS
        gn = float(np.linalg.norm(grad))
        if gn < 1e-9:
            break
        step = ALPHA * grad / gn
        arc = np.linspace(0., 1., RESOLVE_NS); ga = arc[worst_k]
        lo = max(ga - SPAN_PAD, 0.0); hi = min(ga + SPAN_PAD, 1.0)
        for c in range(1, len(cp) - 1):
            g = grev[c]
            if lo <= g <= hi:
                w = win_weight(g, lo, hi)
                cp[c] = np.clip(cp[c] + w * step, POS_LIM[:, 0], POS_LIM[:, 1])
    return cp


def stage_C(cp, names, bases, starts, targets, residual, duration):
    def parked_hits(mover, owner, pose, ns=3 * RESOLVE_NS):
        P = eval_cp_n(cp[mover], ns)
        return sum(1 for k in range(1, ns - 1)
                   if pair_collides(P[k], bases[mover], pose, bases[owner],
                                    margin=CLEAR_M))
    cons = []
    for (na, nb) in residual:
        st = {n: starts[n] for n in (na, nb)}
        tg = {n: targets[n] for n in (na, nb)}
        def attempt(first, second):
            for _ in range(3):
                h1 = parked_hits(first, second, st[second])
                h2 = parked_hits(second, first, tg[first])
                if h1 == 0 and h2 == 0:
                    break
                if h1 > 0:
                    cp[first] = retract_static(cp[first], bases[first],
                                               [st[second]], bases[second])
                if h2 > 0:
                    cp[second] = retract_static(cp[second], bases[second],
                                                [tg[first]], bases[first])
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
            if g1 + g2 < keep[0]:
                first, second = o1[1], o1[0]; h1, h2 = g1, g2
            else:
                cp[na], cp[nb] = keep[1][na], keep[1][nb]
                first, second = keep[2]; h1, h2 = attempt(*keep[2])
        cons.append([first, second])
        print(f'  [C] {na}<->{nb}: order {first} first  parked_hits={h1}+{h2}  '
              f'seq_ok={h1 == 0 and h2 == 0}')
    depth = {n: 0 for n in names}
    for _ in range(len(names)):
        for a, b in cons:
            if depth[b] < depth[a] + 1:
                depth[b] = depth[a] + 1
    for _pass in range(2):
        dirty = False
        for X in names:
            for Y in names:
                if X == Y or depth[X] == depth[Y]:
                    continue
                pose = targets[Y] if depth[Y] < depth[X] else starts[Y]
                h = parked_hits(X, Y, pose)
                if h > 0:
                    dirty = True
                    cp[X] = retract_static(cp[X], bases[X], [pose], bases[Y])
                    h2 = parked_hits(X, Y, pose)
                    kind = 'target' if depth[Y] < depth[X] else 'start'
                    print(f'  [C2] {X} vs {Y}-{kind}: {h} -> {h2} hits')
        if not dirty:
            break
    return cons


def topo_sequential(base_P, names, bases, cons0, duration, res_pairs):
    combos = [cons0]
    for m in range(len(cons0)):
        for bc in list(combos):
            fl = [list(c) for c in bc]
            fl[m] = [fl[m][1], fl[m][0]]
            if fl not in combos and len(combos) < 8:
                combos.append(fl)
    half = max(2, int(round(duration * RATE_HZ)))
    for cons in combos:
        depth = {n: 0 for n in names}
        for _ in range(len(names)):
            for a, b in cons:
                if depth[b] < depth[a] + 1:
                    depth[b] = depth[a] + 1
        K = max(depth.values()) + 1 if depth else 1
        seq = {}
        for n in names:
            p = base_P[n]; blocks = []
            for ph in range(K):
                if ph < depth[n]:
                    blocks.append(np.repeat(p[0][None, :], half, axis=0))
                elif ph == depth[n]:
                    blocks.append(p)
                else:
                    blocks.append(np.repeat(p[-1][None, :], half, axis=0))
            seq[n] = np.vstack(blocks)
        cs_r, _ = mesh_scan(seq, names, bases, stride=4, pairs=res_pairs)
        if cs_r > 0:
            continue
        cs, _ = mesh_scan(seq, names, bases, stride=2)
        if cs == 0:
            return seq, K * duration, K
    return None, None, None

def _write(names, chosen, kdur, base_dur, meta, label, resolved, method):
    out = {'duration': float(kdur), 'arm_names': names,
           'final_duration_s': round(float(kdur), 4),
           'duration_overhead': round(float(kdur) / base_dur - 1, 4),
           'synchronisation_report': {'collision_free': resolved,
                                      'method': method},
           'final_verification': {'verified_collision_free': resolved}}
    for n in names:
        pos = np.asarray(chosen[n]); dt = kdur / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[n] = {'robot_name': n,
                  'metadata': {**meta[n], 'duration': float(kdur)},
                  'trajectory': {'time': np.linspace(0., kdur, len(pos)).tolist(),
                                 'positions': pos.tolist(),
                                 'velocities': vel.tolist(),
                                 'accelerations': acc.tolist(),
                                 'arc_fracs': np.linspace(0., 1., len(pos)).tolist()}}
    with open('s25_synchronized.json', 'w') as fh: json.dump(out, fh)
    print(f'  Saved: s25_synchronized.json ({label})')


def main():
    print('\n' + '=' * 66)
    print('  STEP 25  --  Retraction + Kuramoto RE-ITERATION [6 ARM]')
    print('=' * 66)
    if not os.path.exists('s24_resolved.json'):
        print('  s24_resolved.json not found'); sys.exit(1)
    s24 = json.load(open('s24_resolved.json')); names = s24['arm_names']
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in names}
    if not s24.get('needs_retraction', False):
        out = dict(s24)
        out['final_verification'] = {'verified_collision_free': True,
                                     'reason': 'kuramoto_alone_in_step_24'}
        json.dump(out, open('s25_synchronized.json', 'w'))
        print('  step_24 resolved all by timing -> passthrough.')
        print('  Saved: s25_synchronized.json   Next : step_26\n'); return

    s22 = json.load(open('s22_trajectories.json'))
    duration = float(s22['duration'])
    cp = {n: global_cp_from(s22[n]) for n in names}
    starts = {n: np.array(s22[n]['metadata']['start_joints'], float) for n in names}
    targets = {n: np.array(s22[n]['metadata']['end_joints'], float) for n in names}
    meta = {n: s22[n]['metadata'] for n in names}
    residual0 = [tuple(p) for p in s24.get('residual_pairs', [])]
    print(f'  timing-insufficient pairs from step_24: {residual0}')
    deadline = time.time() + BUDGET_S
    t0 = time.time()
    best = None; best_cs = 10 ** 9; best_dur = duration
    cons = []

    rnd = 0; stale = 0; last_cs = None
    while time.time() < deadline:
        rnd += 1
        paths = {n: eval_cp(cp[n], duration) for n in names}
        residual = residual_of(paths, names, bases)
        if not residual:
            cs, _ = mesh_scan(paths, names, bases, stride=2)
            if cs == 0:
                best, best_cs, best_dur = paths, 0, duration
                break
        for arms, prs in clusters_of(residual):
            upd = resolve_cluster(cp, arms, prs, bases)
            cp.update(upd)
            if len(arms) > 2:
                print(f'  cluster {arms} ({len(prs)} pairs) retracted JOINTLY')
        paths = {n: eval_cp(cp[n], duration) for n in names}
        residual = residual_of(paths, names, bases)
        print(f'  round {rnd}: after retraction residual={residual}  '
              f'({time.time() - t0:.0f}s elapsed)')
        chosen, kdur, cfg = kuramoto_sweep(paths, duration, names, bases,
                                           residual or None, deadline)
        if chosen is not None:
            best, best_cs, best_dur = chosen, 0, kdur
            print(f'  KURAMOTO re-iteration SUCCESS  cfg={cfg}')
            break
        if not residual:
            cs, _ = mesh_scan(paths, names, bases, stride=2)
            if cs == 0:
                best, best_cs, best_dur = paths, 0, duration
                break
        cs, _ = mesh_scan(paths, names, bases, stride=2)
        if cs < best_cs:
            best, best_cs, best_dur = paths, cs, duration
        if rnd >= 2 and residual and time.time() < deadline:
            done = False
            for pair in list(residual):
                for n in pair:
                    saved = cp[n].copy()
                    cp[n] = column_cp(starts[n], targets[n])
                    paths = {m: eval_cp(cp[m], duration) for m in names}
                    res2 = residual_of(paths, names, bases)
                    chosen, kdur, cfg = kuramoto_sweep(paths, duration, names,
                                                       bases, res2 or None,
                                                       deadline)
                    if chosen is not None:
                        best, best_cs, best_dur = chosen, 0, kdur
                        print(f'  [B] rerouted {n}; KURAMOTO SUCCESS  cfg={cfg}')
                        done = True; break
                    if not res2:
                        cs, _ = mesh_scan(paths, names, bases, stride=2)
                        if cs == 0:
                            best, best_cs, best_dur = paths, 0, duration
                            print(f'  [B] rerouted {n}; lockstep clear')
                            done = True; break
                    cp[n] = saved
                    if time.time() > deadline:
                        break
                if done or time.time() > deadline:
                    break
            if done:
                break
            paths = {n: eval_cp(cp[n], duration) for n in names}
            residual = residual_of(paths, names, bases)
            cons = stage_C(cp, names, bases, starts, targets, residual, duration)
            paths = {n: eval_cp(cp[n], duration) for n in names}
            residual = residual_of(paths, names, bases)
            chosen, kdur, cfg = kuramoto_sweep(paths, duration, names, bases,
                                               residual or None, deadline)
            if chosen is not None:
                best, best_cs, best_dur = chosen, 0, kdur
                print(f'  KURAMOTO after parked-pose clearance SUCCESS  cfg={cfg}')
                break
            base_P = {n: resample(paths[n], duration) for n in names}
            seq, sdur, K = topo_sequential(base_P, names, bases, cons, duration,
                                           residual or None)
            if seq is not None:
                best, best_cs, best_dur = seq, 0, sdur
                print(f'  TOPOLOGICAL SEQUENTIAL SUCCESS  (K={K} phases)')
                break
        if last_cs is not None and best_cs >= last_cs:
            stale += 1
            if stale >= NO_PROGRESS_STOP:
                print(f'  no progress in {stale} rounds -> stopping early '
                      f'(structurally unresolvable; {best_cs} coll steps)')
                break
        else:
            stale = 0
        last_cs = best_cs
        if rnd >= 12:
            break

    dev = {}
    for n in names:
        opt = eval_cp(global_cp_from(s22[n]), duration)
        cur = eval_cp(cp[n], duration)
        K = min(len(opt), len(cur))
        dev[n] = round(float(np.mean(np.linalg.norm(cur[:K] - opt[:K], axis=1))), 4)
    touched = {n: v for n, v in dev.items() if v > 1e-6}
    print(f'  deviation from optimal (rad, mean): '
          f'{touched if touched else "none -- geometry untouched"}')

    resolved = best_cs == 0 and best is not None
    if best is None:
        best = {n: eval_cp(cp[n], duration) for n in names}; best_dur = duration
    sc = limit_scale(best, names, best_dur)
    if sc > 1.0:
        best_dur = best_dur * sc * VEL_SAFETY
        best = {n: resample(best[n], best_dur) for n in names}
        print(f'  vel/acc rescale x{sc:.2f} -> duration {best_dur:.2f}s')
    if resolved:
        print(f'  SIMULTANEITY: {100 * simultaneity(best, names):.0f}% of the '
              f'schedule has >1 arm moving')
    print(f'  total resolver time: {time.time() - t0:.0f}s '
          f'(budget {BUDGET_S:.0f}s)')
    method = 'retraction_plus_kuramoto' if resolved else 'unresolved'
    label = ('retraction + Kuramoto SUCCESS' if resolved
             else f'UNRESOLVED ({best_cs} coll steps)')
    _write(names, best, best_dur, duration, meta, label, resolved, method)
    if resolved:
        print(f'  ALL pairs resolved. Duration {duration:.1f}s -> {best_dur:.2f}s')
        print('  Next : ros2 run dual_arm_sync step_26\n')
    else:
        print(f'  Still colliding after budget -> re-run step_21 with other '
              f'branches.\n')


if __name__ == '__main__': main()