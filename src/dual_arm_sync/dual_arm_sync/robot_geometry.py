import numpy as np
from typing import Dict, List, Tuple

_PI   = np.pi
_PI_2 = np.pi / 2.

L1_6 = 0.1525
L2_6 = 0.6200
L3_6 = 0.5590
L4_6 = 0.1210
A_6  = 0.0345

L1 = 0.1525
L2 = 0.620
L3 = 0.22
L4 = 0.195
L5 = 0.14
L6 = 0.121
A1 = 0.21
A2 = 0.1755
A3 = 0.16

DH6 = np.array([
    [0.0,    0.0,   0.0,    L1_6],
    [-_PI_2, 0.0,  -_PI_2,  A_6 ],
    [0.0,    L2_6,  _PI_2,  0.0 ],
    [_PI_2,  0.0,   0.0,    L3_6],
    [-_PI_2, 0.0,   0.0,    0.0 ],
    [_PI_2,  0.0,   0.0,    L4_6],
], dtype=float)

DH9 = np.array([
    [  0.0,    0.0,   0.0,    L1  ],  
    [ -_PI_2,  0.0,  -_PI_2,  A1  ],   
    [  0.0,    L2,    0.0,    0.0 ],  
    [  0.0,    0.0,   _PI_2, -A2  ],   
    [  _PI_2,  0.0,   _PI_2,  L3  ],   
    [  0.0,    A3,    0.0,    L4  ],  
    [  0.0,    0.0,  -_PI_2,  L5  ],   
    [ -_PI_2,  0.0,   0.0,   -A3  ],   
    [  _PI_2,  0.0,   0.0,    L6  ],   
], dtype=float)

JOINT_VAR = np.array([0, 1, -1, 2, 3, -1, -1, 4, 5], dtype=int)

N_FRAMES = 9
NDOF     = 6

POS_LIM = np.array([
    [-2*_PI,  2*_PI ],
    [-1.6493, 1.6493],
    [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.141, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0 ])

CAPSULE_RADII = np.array([
    0.0500,   
    0.0475,  
    0.0450,  
    0.0425,   
    0.0400,  
    0.0375,  
    0.0350,   
    0.0325,   
    0.0300,  
])

SEG_NAMES = ['L1', 'A1', 'L2', 'A2', 'L3', 'L4', 'A3', 'L5', 'L6']

SELF_PAIRS = [
    (0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),
    (1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
    (2,4),(2,5),(2,6),(2,7),(2,8),
    (3,5),(3,6),(3,7),(3,8),
    (4,6),(4,7),(4,8),
    (5,7),(5,8),
    (6,8),
]

INTER_ARM_MARGIN = 0.02  
WARNING_MARGIN   = 0.06  
SELF_MARGIN      = 0.003  


def _dh_mat(al: float, a: float, th: float, d: float) -> np.ndarray:
    ct, st = np.cos(th), np.sin(th)
    ca, sa = np.cos(al), np.sin(al)
    return np.array([
        [ct,    -st,    0.,  a    ],
        [st*ca,  ct*ca, -sa, -sa*d],
        [st*sa,  ct*sa,  ca,  ca*d],
        [0.,     0.,    0.,  1.   ],
    ])

def fk6(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    T = np.eye(4)
    for i in range(6):
        al, a, to, d = DH6[i]
        T = T @ _dh_mat(al, a, q[i] + to, d)
    return T[:3, 3].copy(), T.copy()


def fk6_world(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    pos, _ = fk6(q)
    return pos + np.asarray(base)

def joint_origins_9(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    T = np.eye(4)
    origins = np.zeros((10, 3))
    origins[0] = np.asarray(base).copy()
    b = np.asarray(base)
    for row in range(N_FRAMES):
        al, a, th_off, d = DH9[row]
        jv = JOINT_VAR[row]
        th = th_off + (float(q[jv]) if jv >= 0 else 0.0)
        T  = T @ _dh_mat(al, a, th, d)
        origins[row + 1] = T[:3, 3] + b
    return origins


def fk_ee_9(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    return joint_origins_9(q, base)[-1]


def get_capsules_9(q: np.ndarray,
                   base: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    
    o = joint_origins_9(q, base)
    return [(o[i], o[i+1], CAPSULE_RADII[i]) for i in range(9)]

def seg_seg_dist(p1: np.ndarray, p2: np.ndarray,
                 q1: np.ndarray, q2: np.ndarray) -> float:
    
    d1 = p2 - p1;  d2 = q2 - q1;  r = p1 - q1
    a  = float(np.dot(d1, d1))
    e  = float(np.dot(d2, d2))
    f  = float(np.dot(d2, r))
    EPS = 1e-10
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - q1))
    if a <= EPS:
        s = 0.0;  t = float(np.clip(f / e, 0., 1.))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t = 0.0;  s = float(np.clip(-c / a, 0., 1.))
        else:
            b     = float(np.dot(d1, d2))
            denom = a * e - b * b
            s     = (float(np.clip((b*f - c*e) / denom, 0., 1.))
                     if abs(denom) > EPS else 0.)
            t = (b * s + f) / e
            if t < 0.:
                t = 0.;  s = float(np.clip(-c / a, 0., 1.))
            elif t > 1.:
                t = 1.;  s = float(np.clip((b - c) / a, 0., 1.))
    return float(np.linalg.norm((p1 + s*d1) - (q1 + t*d2)))

def pair_min_dist_9(q_i: np.ndarray, base_i: np.ndarray,
                    q_j: np.ndarray, base_j: np.ndarray) -> float:
    
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    mn = float('inf')
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            d  = seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1]) - ri - rj
            if d < mn:
                mn = d
    return mn


def pair_collides_9(q_i: np.ndarray, base_i: np.ndarray,
                    q_j: np.ndarray, base_j: np.ndarray,
                    margin: float = INTER_ARM_MARGIN) -> bool:
    
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            if seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1]) < ri + rj + margin:
                return True
    return False


def pair_min_dist_9_detail(q_i: np.ndarray, base_i: np.ndarray,
                            q_j: np.ndarray, base_j: np.ndarray,
                            margin: float = INTER_ARM_MARGIN
                            ) -> List[dict]:
    
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    events = []
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            d   = seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1])
            thr = ri + rj + margin
            wrn = ri + rj + WARNING_MARGIN
            if d < wrn:
                events.append({
                    'seg_i'        : i,
                    'seg_j'        : j,
                    'name_i'       : SEG_NAMES[i],
                    'name_j'       : SEG_NAMES[j],
                    'distance_m'   : round(d, 5),
                    'threshold_m'  : round(thr, 5),
                    'penetration_m': round(max(thr - d, 0.0), 5),
                    'collision'    : d < thr,
                    'warning'      : thr <= d < wrn,
                })
    return events

def self_collides_9(q: np.ndarray, base: np.ndarray) -> bool:
    caps = get_capsules_9(q, base)
    for (i, j) in SELF_PAIRS:
        p1, p2, ri = caps[i]
        q1, q2, rj = caps[j]
        if seg_seg_dist(p1, p2, q1, q2) < ri + rj + SELF_MARGIN:
            return True
    return False


def min_self_clearance_9(q: np.ndarray,
                          base: np.ndarray) -> Tuple[float, str]:
    caps   = get_capsules_9(q, base)
    mn_d   = float('inf');  mn_name = ''
    for (i, j) in SELF_PAIRS:
        p1, p2, ri = caps[i]
        q1, q2, rj = caps[j]
        d = seg_seg_dist(p1, p2, q1, q2) - ri - rj
        if d < mn_d:
            mn_d = d;  mn_name = f'{SEG_NAMES[i]}↔{SEG_NAMES[j]}'
    return float(mn_d), mn_name