#!/usr/bin/env python3

import numpy as np


_PI   = np.pi
_PI_2 = np.pi / 2.0
L1, L2, L3, L4 = 0.1525, 0.6200, 0.5590, 0.1210
A = 0.0345

DH = np.array([
    [0.0,    0.0,  0.0,    L1],
    [-_PI_2, 0.0, -_PI_2,  A ],
    [0.0,    L2,   _PI_2,  0.0],
    [_PI_2,  0.0,  0.0,    L3],
    [-_PI_2, 0.0,  0.0,    0.0],
    [_PI_2,  0.0,  0.0,    L4],
], dtype=float)

NDOF    = 6
RATE_HZ = 100.0

POS_LIM = np.array([
    [-2*_PI,  2*_PI ], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0])


ARM_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04', 'dsr05', 'dsr06']

ROBOT_BASES = {
    'dsr01': np.array([ 0.00,  0.725, 0.0]),
    'dsr02': np.array([ 0.00, -0.725, 0.0]),
    'dsr03': np.array([ 1.45,  0.725, 0.0]),
    'dsr04': np.array([ 1.45, -0.725, 0.0]),
    'dsr05': np.array([-1.45,  0.725, 0.0]),
    'dsr06': np.array([-1.45, -0.725, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MARGIN = 0.12
NEAR_MISS_MARGIN = 0.08    
COMFORT_DIST     = 0.30     


DH_TO_CMD = np.array([0, 1, 3, 4, 2, 5])


CAPSULE  = True      
LINK_SUB = 3         


def _chain(q, base=None):
    T = np.eye(4); origins = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0., a],
                          [st*ca, ct*ca, -sa, -sa*d],
                          [st*sa, ct*sa, ca, ca*d],
                          [0., 0., 0., 1.]])
        origins[i] = T[:3, 3] + (base if base is not None else 0.0)
    return T, origins


def fk(q):
    """Return (ee_pos_local, T_4x4)."""
    T, _ = _chain(q)
    return T[:3, 3].copy(), T


def fk_world(q, base):
    return fk(q)[0] + np.asarray(base)


def link_origins(q, base):
    """World-frame origin of each of the 6 links."""
    _, o = _chain(q, np.asarray(base))
    return o


def _seg_seg_dist(p1, q1, p2, q2):
    """Exact minimum distance between segment p1->q1 and segment p2->q2."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-12:
        s = 0.0; t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t = 0.0; s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            bdot = float(d1 @ d2); den = a * e - bdot * bdot
            s = float(np.clip((bdot * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
            t = (bdot * s + f) / e
            if t < 0.0:
                t = 0.0; s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0; s = float(np.clip((bdot - c) / a, 0.0, 1.0))
    cp1 = p1 + d1 * s; cp2 = p2 + d2 * t
    return float(np.linalg.norm(cp1 - cp2))


_L1, _L2, _L3, _L4, _L5, _L6 = 0.1525, 0.620, 0.22, 0.195, 0.14, 0.121
_A1, _A2, _A3 = 0.21, 0.1755, 0.16
_PI_2 = np.pi / 2.0
_DH9 = np.array([
    [  0.0,    0.0,    0.0,   _L1 ],  
    [-_PI_2,   0.0,  -_PI_2,  _A1 ],  
    [  0.0,   _L2,     0.0,    0.0 ],  
    [  0.0,    0.0,   _PI_2, -_A2 ],   
    [ _PI_2,   0.0,   _PI_2,  _L3 ],   
    [  0.0,   _A3,     0.0,   _L4 ],   
    [  0.0,    0.0,  -_PI_2,  _L5 ],   
    [-_PI_2,   0.0,    0.0,  -_A3 ],  
    [ _PI_2,   0.0,    0.0,   _L6 ],   
], dtype=float)
_ACT9 = [0, 1, 3, 4, 7, 8]            
SEG_TO_LINK = [0, 1, 1, 2, 3, 3, 4, 4, 5]
LINK9_RADII = np.array([LINK_RADII[SEG_TO_LINK[s]] for s in range(9)])

def _mdh_T(al, a, th, d):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])

def collision_origins(q, base):
    """10 world-frame origins of the 9-frame chain (accurate link shape)."""
    T = np.eye(4); T[:3, 3] = base
    pts = [T[:3, 3].copy()]
    for i in range(9):
        al, a, th_off, d = _DH9[i]
        th = th_off + (q[_ACT9.index(i)] if i in _ACT9 else 0.0)
        T = T @ _mdh_T(al, a, th, d)
        pts.append(T[:3, 3].copy())
    return np.array(pts)


def _seg_seg_matrix(P1, Q1, P2, Q2):
    """Vectorized: distance matrix [Ni,Nj] between two sets of segments."""
    d1 = Q1 - P1; d2 = Q2 - P2
    r = P1[:, None, :] - P2[None, :, :]
    a = np.sum(d1 * d1, axis=1)[:, None]
    e = np.sum(d2 * d2, axis=1)[None, :]
    f = np.einsum('jk,ijk->ij', d2, r)
    c = np.einsum('ik,ijk->ij', d1, r)
    b = np.einsum('ik,jk->ij', d1, d2)
    den = a * e - b * b
    sden = np.where(den > 1e-12, den, 1.0)
    sa = np.where(a > 1e-12, a, 1.0); se = np.where(e > 1e-12, e, 1.0)
    s0 = np.where(den > 1e-12, np.clip((b * f - c * e) / sden, 0.0, 1.0), 0.0)
    t = (b * s0 + f) / se
    s = np.where(t < 0, np.clip(-c / sa, 0.0, 1.0),
                 np.where(t > 1, np.clip((b - c) / sa, 0.0, 1.0), s0))
    t = np.clip(t, 0.0, 1.0)
    cp1 = P1[:, None, :] + d1[:, None, :] * s[:, :, None]
    cp2 = P2[None, :, :] + d2[None, :, :] * t[:, :, None]
    return np.linalg.norm(cp1 - cp2, axis=2)


def _arm_caps(q, base):
    o = collision_origins(q, base)
    return o, o[:9], o[1:10]


def caps_min_dist(ci, cj):
    """Min centre-line distance from precomputed caps (oi,P1,Q1),(oj,P2,Q2)."""
    oi, P1, Q1 = ci; oj, P2, Q2 = cj
    gap = _aabb_gap(oi, oj)
    if gap > 1.5:
        return gap
    return float(_seg_seg_matrix(P1, Q1, P2, Q2).min())


def caps_collide(ci, cj, margin=SAFETY_MARGIN):
    oi, P1, Q1 = ci; oj, P2, Q2 = cj
    if _aabb_gap(oi, oj) > 2 * LINK9_RADII.max() + margin:
        return False
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    thr = LINK9_RADII[:, None] + LINK9_RADII[None, :] + margin
    return bool((M < thr).any())


def caps_surface_dist(ci, cj):
    oi, P1, Q1 = ci; oj, P2, Q2 = cj
    gap = _aabb_gap(oi, oj)
    if gap > 1.5:
        return gap - 2.0 * float(LINK9_RADII.max())
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    return float((M - (LINK9_RADII[:, None] + LINK9_RADII[None, :])).min())


def link_segments(q, base):
    """9 capsules (p, q, radius, seg_index) along the 9-frame chain."""
    o = collision_origins(q, base)
    if not CAPSULE:
        return [(o[i], o[i], LINK9_RADII[min(i, 8)], i) for i in range(len(o))]
    return [(o[s], o[s + 1], LINK9_RADII[s], s) for s in range(9)]


def _aabb_gap(oi, oj):
    """Positive lower-bound on the gap between two arms' link-origin boxes."""
    lo = np.maximum(oi.min(0), oj.min(0)); hi = np.minimum(oi.max(0), oj.max(0))
    d = lo - hi
    return float(np.linalg.norm(np.maximum(d, 0.0)))   # 0 if boxes overlap


def pair_min_dist(qi, bi, qj, bj):
    
    oi, P1, Q1 = _arm_caps(qi, bi); oj, P2, Q2 = _arm_caps(qj, bj)
    gap = _aabb_gap(oi, oj)
    if gap > 1.5:
        return gap - 2.0 * float(LINK9_RADII.max())
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    return float((M - (LINK9_RADII[:, None] + LINK9_RADII[None, :])).min())


def pair_collides(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    oi, P1, Q1 = _arm_caps(qi, bi); oj, P2, Q2 = _arm_caps(qj, bj)
    if _aabb_gap(oi, oj) > 2 * LINK9_RADII.max() + margin:
        return False
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    thr = LINK9_RADII[:, None] + LINK9_RADII[None, :] + margin
    return bool((M < thr).any())


def deepest_link_pair(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    """(link_i, link_j, penetration_m) mapped to 6-DOF link indices, or None."""
    oi, P1, Q1 = _arm_caps(qi, bi); oj, P2, Q2 = _arm_caps(qj, bj)
    if _aabb_gap(oi, oj) > 2 * LINK9_RADII.max() + margin:
        return None
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    pen = (LINK9_RADII[:, None] + LINK9_RADII[None, :] + margin) - M
    idx = np.unravel_index(int(np.argmax(pen)), pen.shape)
    if pen[idx] <= 0:
        return None
    return (SEG_TO_LINK[idx[0]], SEG_TO_LINK[idx[1]], float(pen[idx]))



def truncated_jacobian_linear(q, link_idx, base, eps=1e-6):
    
    J = np.zeros((3, NDOF))
    o0 = link_origins(q, base)[link_idx]
    for j in range(link_idx + 1):
        dq = q.copy(); dq[j] += eps
        oj = link_origins(dq, base)[link_idx]
        J[:, j] = (oj - o0) / eps
    return J



def _mesh_warning(reason):
   
    import os as _o, sys as _s
    bar = '!' * 74
    msg = (f"\n{bar}\n"
           f"  MESH COLLISION UNAVAILABLE -- FALLING BACK TO CAPSULES\n"
           f"  reason: {reason}\n"
           f"  Results will NOT match the mesh pipeline: capsules are\n"
           f"  shape-blind and roll-blind, and under-represent parts such as\n"
           f"  MF1013_2_1 by ~21 cm of bounding radius.\n"
           f"  FIX:  pip install python-fcl trimesh pycollada\n"
           f"        python3 mesh_collision.py --build /path/to/ros2_ws\n"
           f"  (set DUAL_ARM_REQUIRE_MESH=1 to abort instead of degrading)\n"
           f"{bar}\n")
    print(msg, file=_s.stderr)
    if _o.environ.get('DUAL_ARM_REQUIRE_MESH'):
        raise SystemExit('aborting: DUAL_ARM_REQUIRE_MESH=1 and meshes are unavailable')


USE_MESH = True
if USE_MESH:
    try:
        import mesh_collision as _mc
        if _mc.available():
            pair_collides     = _mc.pair_collides      
            pair_min_dist     = _mc.pair_min_dist       
            deepest_link_pair = _mc.deepest_link_pair    
            print("[_robot8x] REAL-MESH collision active (FCL)")
        else:
            _mesh_warning('mesh_collision reported unavailable')
    except Exception as _e:
        _mesh_warning('import failed (%s)' % _e)




def _chain_frames(q, base):
    """World 4x4 transform of each of the 6 DH links."""
    T = np.eye(4); out = []
    b = np.asarray(base, float)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0., a],
                          [st*ca, ct*ca, -sa, -sa*d],
                          [st*sa, ct*sa, ca, ca*d],
                          [0., 0., 0., 1.]])
        Tw = T.copy(); Tw[:3, 3] = Tw[:3, 3] + b
        out.append(Tw)
    return out


def _seg_seg_points(p1, q1, p2, q2):
    """Closest POINTS (cp1, cp2) between segment p1->q1 and p2->q2."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return p1.copy(), p2.copy()
    if a <= 1e-12:
        s = 0.0; t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t = 0.0; s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            bdot = float(d1 @ d2); den = a * e - bdot * bdot
            s = float(np.clip((bdot * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
            t = (bdot * s + f) / e
            if t < 0.0:
                t = 0.0; s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0; s = float(np.clip((bdot - c) / a, 0.0, 1.0))
    return p1 + d1 * s, p2 + d2 * t


def witness_pair(qi, bi, qj, bj):
    """Deepest capsule contact. n_hat points FROM arm j TOWARD arm i.
    dist = centre-line distance minus radii (negative = overlap depth)."""
    oi, P1, Q1 = _arm_caps(qi, np.asarray(bi))
    oj, P2, Q2 = _arm_caps(qj, np.asarray(bj))
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    sep = M - (LINK9_RADII[:, None] + LINK9_RADII[None, :])
    a, b = np.unravel_index(int(np.argmin(sep)), sep.shape)
    cp_i, cp_j = _seg_seg_points(P1[a], Q1[a], P2[b], Q2[b])
    v = cp_i - cp_j; nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        v = oi[a] - oj[b]; nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            v = np.array([0.0, 0.0, 1.0]); nv = 1.0
    return {'cp_i': cp_i, 'cp_j': cp_j,
            'link_i': int(SEG_TO_LINK[a]), 'link_j': int(SEG_TO_LINK[b]),
            'seg_i': int(a), 'seg_j': int(b),
            'dist': float(sep[a, b]), 'n_hat': v / nv}


def point_jacobian(q, base, link_idx, p_world, eps=1e-6):
    """(3, NDOF) linear Jacobian of p_world rigidly attached to link link_idx.
    Downstream columns are zero."""
    q = np.asarray(q, float)
    frames = _chain_frames(q, base)
    Ti = frames[link_idx]
    Ri = Ti[:3, :3]; ti = Ti[:3, 3]
    p_local = Ri.T @ (np.asarray(p_world, float) - ti)
    J = np.zeros((3, NDOF))
    for k in range(link_idx + 1):
        dq = q.copy(); dq[k] += eps
        Tk = _chain_frames(dq, base)[link_idx]
        p_new = Tk[:3, :3] @ p_local + Tk[:3, 3]
        J[:, k] = (p_new - np.asarray(p_world, float)) / eps
    return J


def escape_dq(qi, bi, qj, bj, alpha=1.0, both=True):
    """dq = alpha * J^T * n_hat at the witness points; arm i pushed along
    +n_hat, arm j along -n_hat. Returns (dq_i, dq_j, info)."""
    w = witness_pair(qi, bi, qj, bj)
    n = w['n_hat']
    Ji = point_jacobian(qi, bi, w['link_i'], w['cp_i'])
    dq_i = alpha * (Ji.T @ n)
    if both:
        Jj = point_jacobian(qj, bj, w['link_j'], w['cp_j'])
        dq_j = alpha * (Jj.T @ (-n))
    else:
        dq_j = np.zeros(NDOF)
    return dq_i, dq_j, w


_MESH_WITNESS = False
if USE_MESH:
    try:
        import mesh_collision as _mcw
        if _mcw.available():
            _MESH_WITNESS = True
    except Exception:
        _MESH_WITNESS = False

_capsule_witness_pair = witness_pair
_capsule_escape_dq = escape_dq


def witness_pair_mesh(qi, bi, qj, bj):
    """Mesh-surface witness; falls back to the capsule witness only when the
    meshes interpenetrate. Return keys match the capsule version, plus
    'source' ('mesh' or 'capsule') and, for mesh hits, 'frame_i'/'frame_j'."""
    if _MESH_WITNESS:
        w = _mcw.mesh_witness_pair(qi, bi, qj, bj)
        if w.get('ok'):
            w['source'] = 'mesh'
            return w
    w = _capsule_witness_pair(qi, bi, qj, bj)
    w['source'] = 'capsule'
    return w


def escape_dq_mesh(qi, bi, qj, bj, alpha=1.0, both=True):
    """Perpendicular escape driven by the MESH surface witness:

        dq_i = +alpha * J_i(cp_i)^T * n_hat
        dq_j = -alpha * J_j(cp_j)^T * n_hat

    BOTH arms move by default. Measured on the real meshes: two-sided escape
    cleared a 1.6cm contact moving arm1 by 2.35 deg, where the one-sided
    escape needed 5.15 deg of arm1 -- i.e. each arm stays roughly half as far
    from its step_82 optimal path when the pair shares the burden.
    Under interpenetration this degrades to the capsule direction."""
    w = witness_pair_mesh(qi, bi, qj, bj)
    if w.get('source') == 'mesh':
        Ji = _mcw.point_jacobian_urdf(qi, bi, w['frame_i'], w['cp_i'])
        dq_i = alpha * (Ji.T @ w['n_hat'])
        if both:
            Jj = _mcw.point_jacobian_urdf(qj, bj, w['frame_j'], w['cp_j'])
            dq_j = alpha * (Jj.T @ (-w['n_hat']))
        else:
            dq_j = np.zeros(NDOF)
        return dq_i, dq_j, w
    return _capsule_escape_dq(qi, bi, qj, bj, alpha=alpha, both=both)


if _MESH_WITNESS:
    witness_pair = witness_pair_mesh      # noqa: F811
    escape_dq = escape_dq_mesh            # noqa: F811
    print("[_robot8x] MESH witness/escape active (surface points, URDF frames)")



import os as _os

CLEAR_M     = float(_os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
DEG         = 3
SEED_NCP    = 12
RESOLVE_NS  = 220
SCAN_STRIDE = int(_os.environ.get('DUAL_ARM_SCAN_STRIDE', '2'))

PREFILTER_M = float(_os.environ.get('DUAL_ARM_PREFILTER_M', '0.05'))

CAPS_BOUND_SLACK = float(_os.environ.get('DUAL_ARM_CAPS_SLACK', '0.35'))
try:
    import mesh_collision as _mcw2
    _MESH_BOUND = _mcw2.available()
except Exception:
    _MESH_BOUND = False

from scipy.interpolate import BSpline as _BSpline

def make_knots(ncp):
    ni = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def basis_row(ncp, u):
    kn = make_knots(ncp)
    u = float(np.clip(u, 0.0, 1.0 - 1e-9))
    return np.asarray(_BSpline.design_matrix([u], kn, DEG).todense()).ravel()


def eval_cp_n(cp, n):
    kn = make_knots(len(cp)); s = np.linspace(0., 1., n)
    P = np.zeros((n, NDOF))
    for j in range(NDOF):
        P[:, j] = _BSpline(kn, cp[:, j], DEG, extrapolate=True)(s)
    return np.clip(P, POS_LIM[:, 0], POS_LIM[:, 1])


def eval_cp(cp, duration):
    return eval_cp_n(cp, max(2, int(round(duration * RATE_HZ))))


def resample(pos, duration):
    nout = max(2, int(round(duration * RATE_HZ)))
    sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
    return np.clip(np.vstack([np.interp(sout, sin, pos[:, j])
                              for j in range(NDOF)]).T,
                   POS_LIM[:, 0], POS_LIM[:, 1])


def interp(path, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(path) - 1
    if n <= 0: return path[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return path[idx] + a * (path[min(idx + 1, n)] - path[idx])



def _caps_surface(qi, bi, qj, bj):
    
    if _MESH_BOUND:
        return _mcw.broad_lower_bound(qi, bi, qj, bj)
    _, P1, Q1 = _arm_caps(qi, bi)
    _, P2, Q2 = _arm_caps(qj, bj)
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    return float((M - (LINK9_RADII[:, None] + LINK9_RADII[None, :])).min()) \
        - CAPS_BOUND_SLACK


def mesh_scan(P, names, bases, stride=None, pairs=None, gate=None):
   
    if stride is None: stride = SCAN_STRIDE
    if gate is None: gate = CLEAR_M
    if pairs is None:
        pairs = [(names[i], names[j]) for i in range(len(names))
                 for j in range(i + 1, len(names))]
    cs = 0; mind = float('inf')
    for ni, nj in pairs:
        K = min(len(P[ni]), len(P[nj]))
        bi, bj = bases[ni], bases[nj]
        for k in range(0, K, stride):
            if _caps_surface(P[ni][k], bi, P[nj][k], bj) > PREFILTER_M:
                continue
            d = pair_min_dist(P[ni][k], bi, P[nj][k], bj)
            if d < mind: mind = d
            if d < gate: cs += 1
    return cs, (mind if mind != float('inf') else 1.0)


def residual_of(P, names, bases, stride=None):
    
    res = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c, _ = mesh_scan(P, names, bases, stride=stride,
                             pairs=[(names[i], names[j])])
            if c > 0:
                res.append((names[i], names[j]))
    return res


def limit_scale(P, names, duration, compress=False):
    s = 0.0 if compress else 1.0
    for n in names:
        pos = P[n]; dt = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        for j in range(NDOF):
            vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
            if vp > VEL_LIM[j]: s = max(s, vp / VEL_LIM[j])
            if ap > ACC_LIM[j]: s = max(s, float(np.sqrt(ap / ACC_LIM[j])))
    return s


def simultaneity(P, names):
   
    K = min(len(P[n]) for n in names)
    if K < 3: return 1.0
    mov = np.zeros((K - 1, len(names)), bool)
    for i, n in enumerate(names):
        mov[:, i] = np.max(np.abs(np.diff(np.asarray(P[n])[:K], axis=0)), axis=1) > 1e-5
    nm = mov.sum(axis=1); act = nm >= 1
    return float((nm[act] > 1).mean()) if act.any() else 1.0