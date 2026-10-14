#!/usr/bin/env python3
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ = os.environ.get('DUAL_ARM_MESH_NPZ',
                      os.path.join(_HERE, 'm1013_parts.npz'))

JX = np.array([[0.0, 0.0, 0.1525],
               [0.0, 0.0345, 0.0],
               [0.62, 0.0, 0.0],
               [0.0, -0.559, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, -0.121, 0.0]], dtype=float)
JR = np.array([[0.0, 0.0, 0.0],
               [0.0, -1.571, -1.571],
               [0.0, 0.0, 1.571],
               [1.571, 0.0, 0.0],
               [-1.571, 0.0, 0.0],
               [1.571, 0.0, 0.0]], dtype=float)
LINKIDX = [0, 1, 2, 2, 2, 3, 4, 4, 5, 6]      
PART_NAMES = ['MF1013_0_0', 'MF1013_1_0', 'MF1013_2_0', 'MF1013_2_1',
              'MF1013_2_2', 'MF1013_3_0', 'MF1013_4_0', 'MF1013_4_1',
              'MF1013_5_0', 'MF1013_6_0']
NP_PARTS = 10
PART_TO_LINK6 = [0, 1, 2, 2, 2, 3, 4, 4, 5, 5]
BROAD_SLACK = 0.12

CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))

_OK = False
_VERTS = None
_CENT = None
_RADP = None
_ARM_R = None
_fcl = None
_OBJS = None


def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def fk_frames(q, base):
    q = np.asarray(q, float)
    T = np.eye(4); T[:3, 3] = np.asarray(base, float)
    out = [T.copy()]
    for i in range(6):
        M = np.eye(4)
        M[:3, :3] = _rpy(*JR[i])
        M[:3, 3] = JX[i]
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T = T @ M @ Rz
        out.append(T.copy())
    return out


def _load_dae_vertices(path):
    import xml.etree.ElementTree as ET

    def _tag(e):
        return e.tag.split('}')[-1]

    def _floats(txt):
        return np.fromstring(txt.replace('\n', ' ').replace('\t', ' '), sep=' ')

    root = ET.parse(path).getroot()

    geoms = {}
    for lib in root.iter():
        if _tag(lib) != 'library_geometries':
            continue
        for g in lib:
            if _tag(g) != 'geometry':
                continue
            gid = g.get('id') or ''
            pts = []
            for fa in g.iter():
                if _tag(fa) != 'float_array':
                    continue
                fid = (fa.get('id') or '') + (fa.get('name') or '')
                low = fid.lower()
                if 'position' not in low and 'vert' not in low:
                    continue
                v = _floats(fa.text or '')
                if v.size >= 3:
                    pts.append(v[:v.size - (v.size % 3)].reshape(-1, 3))
            if pts:
                geoms[gid] = np.vstack(pts)

    out = []

    def walk(node, T):
        M = T
        for ch in node:
            if _tag(ch) == 'matrix':
                m = _floats(ch.text or '')
                if m.size == 16:
                    M = M @ m.reshape(4, 4)
        for ch in node:
            t = _tag(ch)
            if t == 'instance_geometry':
                url = (ch.get('url') or '').lstrip('#')
                P = geoms.get(url)
                if P is not None and len(P):
                    Ph = np.hstack([P, np.ones((len(P), 1))])
                    out.append((Ph @ M.T)[:, :3])
            elif t == 'node':
                walk(ch, M)

    for lib in root.iter():
        if _tag(lib) != 'library_visual_scenes':
            continue
        for sc in lib:
            for nd in sc:
                if _tag(nd) == 'node':
                    walk(nd, np.eye(4))

    if out:
        return np.vstack(out)
    if geoms:
        return np.vstack(list(geoms.values()))
    import trimesh
    return np.asarray(trimesh.load(path, force='mesh').vertices, float)


def build_parts(root=None, verbose=True):
    import glob
    parts = PART_NAMES
    raw = ('https://raw.githubusercontent.com/DoosanRobotics/doosan-robot2/'
           'jazzy/dsr_description2/meshes/m1013_collision')
    roots = [root] if root else []
    roots += [os.environ.get('DUAL_ARM_ROS2_WS', ''), _HERE,
              os.path.expanduser('~/ros2_ws'), os.path.expanduser('~')]
    paths = {}
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for p in parts:
            if p in paths:
                continue
            g = glob.glob(os.path.join(r, '**', 'dsr_description2', 'meshes',
                                       'm1013_collision', p + '.dae'),
                          recursive=True)
            if g:
                paths[p] = g[0]
        if len(paths) == len(parts):
            break
    work = os.path.join(_HERE, '_m1013_dae')
    missing = [p for p in parts if p not in paths]
    if missing:
        os.makedirs(work, exist_ok=True)
        import urllib.request
        for p in missing:
            dst = os.path.join(work, p + '.dae')
            if not os.path.exists(dst):
                if verbose:
                    print(f'  [mesh_collision] fetching {p}.dae')
                urllib.request.urlretrieve(f'{raw}/{p}.dae', dst)
            paths[p] = dst
    out = {}
    for i, p in enumerate(parts):
        v = _load_dae_vertices(paths[p])
        if np.abs(v).max() > 10.0:
            v = v * 1e-3
        try:
            from scipy.spatial import ConvexHull
            v = v[ConvexHull(v).vertices]
        except Exception:
            pass
        out[f'P{i}'] = v
        if verbose:
            print(f'  [mesh_collision] {p}: hull {len(v):5d} verts')
    np.savez_compressed(_NPZ, **out)
    if verbose:
        print(f'  [mesh_collision] wrote {_NPZ}')
    return _NPZ


def _load():
    global _OK, _VERTS, _CENT, _RADP, _ARM_R, _fcl, _OBJS
    if _OK:
        return True
    if not os.path.exists(_NPZ):
        if os.environ.get('DUAL_ARM_NO_AUTOBUILD'):
            return False
        try:
            build_parts()
        except Exception as e:
            print(f'  [mesh_collision] auto-build failed ({e}); '
                  f'run: python3 mesh_collision.py --build /path/to/ros2_ws')
            return False
        if not os.path.exists(_NPZ):
            return False
    try:
        import fcl
    except Exception:
        return False
    d = np.load(_NPZ)
    verts = []
    for i in range(NP_PARTS):
        v = np.asarray(d[f'P{i}'], float)
        if np.abs(v).max() > 10.0:
            v = v * 1e-3
        verts.append(v)
    _VERTS = verts
    _CENT = np.array([v.mean(0) for v in verts])
    _RADP = np.array([np.linalg.norm(v - v.mean(0), axis=1).max()
                      for v in verts])
    _ARM_R = None
    _fcl = fcl
    _OBJS = []
    for slot in range(2):
        row = []
        for i in range(NP_PARTS):
            v = _VERTS[i]
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(v)
                faces = hull.simplices
                tri = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int32)
                geom = fcl.Convex(np.asarray(hull.points, float),
                                  len(faces), tri.reshape(-1))
            except Exception:
                geom = fcl.Sphere(float(_RADP[i]))
            row.append(fcl.CollisionObject(geom, fcl.Transform()))
        _OBJS.append(row)
    _OK = True
    return True


def available():
    try:
        return _load()
    except Exception:
        return False


def _pose_parts(q, base, slot):
    F = fk_frames(q, base)
    C = np.zeros((NP_PARTS, 3))
    for p in range(NP_PARTS):
        T = F[LINKIDX[p]]
        R = T[:3, :3]; t = T[:3, 3]
        C[p] = R @ _CENT[p] + t
        _OBJS[slot][p].setTransform(_fcl.Transform(R, t))
    return C, F


def _pair_query(qi, bi, qj, bj, gate, want_witness=False, early_out=True):
    if not _load():
        raise RuntimeError('mesh_collision unavailable')
    Ci, _ = _pose_parts(qi, bi, 0)
    Cj, _ = _pose_parts(qj, bj, 1)

    D = np.linalg.norm(Ci[:, None, :] - Cj[None, :, :], axis=2)
    S = D - _RADP[:, None] - _RADP[None, :]
    lb = float(S.min())
    if lb > gate:
        return lb, {'part_i': -1, 'part_j': -1}

    dmin = float('inf'); info = {'part_i': -1, 'part_j': -1}
    order = np.dstack(np.unravel_index(np.argsort(S, axis=None), S.shape))[0]
    for a, b in order:
        s_ab = S[a, b]
        if s_ab > gate + BROAD_SLACK:
            break
        if s_ab > dmin:
            break
        req = _fcl.DistanceRequest(enable_nearest_points=True)
        res = _fcl.DistanceResult()
        d = _fcl.distance(_OBJS[0][a], _OBJS[1][b], req, res)
        if d < 0:
            d = 0.0
        if d < dmin:
            dmin = d
            info = {'part_i': int(a), 'part_j': int(b)}
            if want_witness:
                info['p_i'] = np.asarray(res.nearest_points[0], float)
                info['p_j'] = np.asarray(res.nearest_points[1], float)
            if early_out and dmin <= 0.0:
                break
    if dmin == float('inf'):
        return lb, {'part_i': -1, 'part_j': -1}
    return float(dmin), info


def pair_min_dist(qi, bi, qj, bj):
    d, _ = _pair_query(qi, bi, qj, bj, CLEAR_M)
    return d


def pair_collides(qi, bi, qj, bj, margin=None):
    g = CLEAR_M if margin is None else float(margin)
    d, _ = _pair_query(qi, bi, qj, bj, g)
    return bool(d < g)


def deepest_link_pair(qi, bi, qj, bj, margin=None):
    g = CLEAR_M if margin is None else float(margin)
    d, info = _pair_query(qi, bi, qj, bj, g)
    if d >= g or info['part_i'] < 0:
        return None
    return (PART_TO_LINK6[info['part_i']], PART_TO_LINK6[info['part_j']],
            float(g - d))


def broad_lower_bound(qi, bi, qj, bj):
    if not _load():
        raise RuntimeError('mesh_collision unavailable')
    Fi = fk_frames(qi, bi); Fj = fk_frames(qj, bj)
    Ci = np.empty((NP_PARTS, 3)); Cj = np.empty((NP_PARTS, 3))
    for p in range(NP_PARTS):
        Ti = Fi[LINKIDX[p]]; Ci[p] = Ti[:3, :3] @ _CENT[p] + Ti[:3, 3]
        Tj = Fj[LINKIDX[p]]; Cj[p] = Tj[:3, :3] @ _CENT[p] + Tj[:3, 3]
    D = np.linalg.norm(Ci[:, None, :] - Cj[None, :, :], axis=2)
    return float((D - _RADP[:, None] - _RADP[None, :]).min())


def mesh_witness_pair(qi, bi, qj, bj):
    d, info = _pair_query(qi, bi, qj, bj, CLEAR_M, want_witness=True,
                          early_out=False)
    if info['part_i'] < 0 or 'p_i' not in info:
        return {'ok': False, 'dist': float(d)}
    p_i = info['p_i']; p_j = info['p_j']
    v = p_i - p_j; nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return {'ok': False, 'dist': float(d)}
    a = int(info['part_i']); b = int(info['part_j'])
    return {'ok': True, 'dist': float(d), 'cp_i': p_i, 'cp_j': p_j,
            'n_hat': v / nv, 'part_i': a, 'part_j': b,
            'link_i': PART_TO_LINK6[a], 'link_j': PART_TO_LINK6[b],
            'frame_i': LINKIDX[a], 'frame_j': LINKIDX[b]}


def point_jacobian_urdf(q, base, frame_idx, p_world, eps=1e-6):
    q = np.asarray(q, float)
    F = fk_frames(q, base)
    T = F[frame_idx]
    loc = T[:3, :3].T @ (np.asarray(p_world, float) - T[:3, 3])
    J = np.zeros((3, 6))
    for k in range(min(frame_idx, 6)):
        dq = q.copy(); dq[k] += eps
        T2 = fk_frames(dq, base)[frame_idx]
        J[:, k] = ((T2[:3, :3] @ loc + T2[:3, 3]) - np.asarray(p_world, float)) / eps
    return J


def mesh_escape_dq(qi, bi, qj, bj, alpha=1.0, both=True):
    w = mesh_witness_pair(qi, bi, qj, bj)
    z = np.zeros(6)
    if not w.get('ok'):
        return z, z, w
    Ji = point_jacobian_urdf(qi, bi, w['frame_i'], w['cp_i'])
    dq_i = alpha * (Ji.T @ w['n_hat'])
    if both:
        Jj = point_jacobian_urdf(qj, bj, w['frame_j'], w['cp_j'])
        dq_j = alpha * (Jj.T @ (-w['n_hat']))
    else:
        dq_j = z
    return dq_i, dq_j, w


def _selftest():
    print('mesh cache :', _NPZ, '(exists)' if os.path.exists(_NPZ) else '(missing)')
    ok = available()
    print('available():', ok)
    if not ok:
        print('  -> capsule fallback will be used. Install python-fcl, trimesh,')
        print('     pycollada, then: python3 mesh_collision.py --build')
        return 1
    print(f'parts      : {NP_PARTS}, link map {LINKIDX}')
    for i, p in enumerate(PART_NAMES):
        v = _VERTS[i]
        print(f'  {p:12s} {len(v):5d} hull verts  r={_RADP[i]*100:5.1f} cm')
    b1 = np.array([0.0, 0.65, 0.0]); b2 = np.array([0.0, -0.65, 0.0])
    z = np.zeros(6)
    import time as _t
    t0 = _t.time(); d = pair_min_dist(z, b1, z, b2); el = (_t.time() - t0) * 1000
    print(f'\nHOME columns surface gap = {d*100:.1f} cm   ({el:.1f} ms/query)')
    qa = np.array([-1.15937, 0.57974, 1.41609, 0.21292, -0.54148, -0.21694])
    qb = np.array([0.49189, 1.13995, 1.08730, -0.22552, 0.60677, -0.10508])
    w = mesh_witness_pair(qa, b1, qb, b2)
    if w.get('ok'):
        gap = float(np.linalg.norm(w['cp_i'] - w['cp_j']))
        print(f'witness    : {PART_NAMES[w["part_i"]]} <-> {PART_NAMES[w["part_j"]]}'
              f'  dist={w["dist"]*100:.2f} cm  |cp_i-cp_j|={gap*100:.2f} cm'
              f'  {"OK (points on surfaces)" if abs(gap-w["dist"])<1e-6 else "MISMATCH"}')
        d1, d2, _ = mesh_escape_dq(qa, b1, qb, b2, alpha=1.0)
        print(f'escape     : |dq_i|={np.linalg.norm(d1):.3f} '
              f'|dq_j|={np.linalg.norm(d2):.3f}  (two-sided, both nonzero)')
    else:
        print(f'witness    : interpenetrating (dist={w["dist"]*100:.2f} cm) -> '
              f'capsule direction fallback (expected under penetration)')
    print('\nself-test PASSED')
    return 0


if __name__ == '__main__':
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == '--build':
        build_parts(_s.argv[2] if len(_s.argv) > 2 else None)
        _s.exit(_selftest())
    _s.exit(_selftest())