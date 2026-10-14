
import numpy as np
from typing import Dict, List, Tuple, Optional

ROBOT_BASES: Dict[str, np.ndarray] = {
    "dsr01": np.array([0.0,  0.5, 0.0]),
    "dsr02": np.array([0.0, -0.5, 0.0]),
    "dsr03": np.array([1.0,  0.5, 0.0]),
    "dsr04": np.array([1.0, -0.5, 0.0]),
}

ARM_NAMES: List[str] = ["dsr01", "dsr02", "dsr03", "dsr04"]

INTER_ARM_PAIRS: List[Tuple[str, str]] = [
    ("dsr01", "dsr02"),
    ("dsr01", "dsr03"),
    ("dsr01", "dsr04"),
    ("dsr02", "dsr03"),
    ("dsr02", "dsr04"),
    ("dsr03", "dsr04"),
]

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
    [ 0.0,    0.0,   0.0,    L1 ],  
    [-_PI_2,  0.0,  -_PI_2,  A1 ],  
    [ 0.0,    L2,    0.0,    0.0], 
    [ 0.0,    0.0,   _PI_2, -A2 ],  
    [ _PI_2,  0.0,   _PI_2,  L3 ],  
    [ 0.0,    A3,    0.0,    L4 ],  
    [ 0.0,    0.0,  -_PI_2,  L5 ],  
    [-_PI_2,  0.0,   0.0,   -A3 ], 
    [ _PI_2,  0.0,   0.0,    L6 ],  
], dtype=float)

JOINT_VAR = np.array([0, 1, -1, 2, 3, -1, -1, 4, 5], dtype=int)

N_FRAMES = 9
NDOF     = 6

POS_LIM = np.array([
    [-2*_PI,   2*_PI  ],
    [-1.6493,  1.6493 ],
    [-2.7925,  2.7925 ],
    [-2*_PI,   2*_PI  ],
    [-2*_PI,   2*_PI  ],
    [-2*_PI,   2*_PI  ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.141, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,   12.0,  12.0,  12.0 ])

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

INTER_ARM_MARGIN = 0.20  
ENDPOINT_MARGIN  = 0.05  
SELF_MARGIN      = 0.003 

def fk6(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    for i in range(6):
        al, a, to, d = DH6[i]
        th = q[i] + to
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a   ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.  ],
        ])
    return T[:3, 3].copy(), T.copy()


def fk6_world(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    pos, _ = fk6(q)
    return pos + np.asarray(base)


def fk6_world_all(qs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    
    return {name: fk6_world(qs[name], ROBOT_BASES[name]) for name in ARM_NAMES}

def joint_origins_9(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    T = np.eye(4)
    origins = np.zeros((10, 3))
    origins[0] = np.asarray(base).copy()
    for row in range(N_FRAMES):
        al, a, th_off, d = DH9[row]
        jv = JOINT_VAR[row]
        th = th_off + (float(q[jv]) if jv >= 0 else 0.0)
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a   ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.  ],
        ])
        origins[row + 1] = T[:3, 3] + np.asarray(base)
    return origins


def fk_ee_9(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    return joint_origins_9(q, base)[-1]


def get_capsules_9(
    q: np.ndarray,
    base: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
   
    origins = joint_origins_9(q, base)
    return [(origins[i], origins[i+1], CAPSULE_RADII[i]) for i in range(9)]

def seg_seg_dist(
    p1: np.ndarray, p2: np.ndarray,
    q1: np.ndarray, q2: np.ndarray,
) -> float:
    d1 = p2-p1; d2 = q2-q1; r = p1-q1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    EPS = 1e-10
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - q1))
    if a <= EPS:
        s = 0.0; t = float(np.clip(f/e, 0., 1.))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t = 0.0; s = float(np.clip(-c/a, 0., 1.))
        else:
            b     = float(np.dot(d1, d2))
            denom = a*e - b*b
            s = float(np.clip((b*f - c*e)/denom, 0., 1.)) if abs(denom) > EPS else 0.
            t = (b*s + f) / e
            if t < 0.:
                t = 0.; s = float(np.clip(-c/a, 0., 1.))
            elif t > 1.:
                t = 1.; s = float(np.clip((b-c)/a, 0., 1.))
    return float(np.linalg.norm((p1+s*d1) - (q1+t*d2)))

def pair_min_dist_9(
    q_i: np.ndarray, base_i: np.ndarray,
    q_j: np.ndarray, base_j: np.ndarray,
) -> float:
    """Min capsule-surface distance between two arms. Negative = interpenetration."""
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


def pair_collides_9(
    q_i: np.ndarray, base_i: np.ndarray,
    q_j: np.ndarray, base_j: np.ndarray,
    margin: float = INTER_ARM_MARGIN,
) -> bool:
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            if seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1]) < ri + rj + margin:
                return True
    return False

def all_pairs_min_dist(
    qs: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], float]:
    
    result: Dict[Tuple[str, str], float] = {}
    for (ni, nj) in INTER_ARM_PAIRS:
        result[(ni, nj)] = pair_min_dist_9(
            qs[ni], ROBOT_BASES[ni],
            qs[nj], ROBOT_BASES[nj],
        )
    return result


def any_pair_collides(
    qs: Dict[str, np.ndarray],
    margin: float = INTER_ARM_MARGIN,
) -> bool:
    
    for (ni, nj) in INTER_ARM_PAIRS:
        if pair_collides_9(
            qs[ni], ROBOT_BASES[ni],
            qs[nj], ROBOT_BASES[nj],
            margin=margin,
        ):
            return True
    return False


def colliding_pairs(
    qs: Dict[str, np.ndarray],
    margin: float = INTER_ARM_MARGIN,
) -> List[Tuple[str, str]]:
    
    bad: List[Tuple[str, str]] = []
    for (ni, nj) in INTER_ARM_PAIRS:
        if pair_collides_9(
            qs[ni], ROBOT_BASES[ni],
            qs[nj], ROBOT_BASES[nj],
            margin=margin,
        ):
            bad.append((ni, nj))
    return bad



def self_collides_9(q: np.ndarray, base: np.ndarray) -> bool:
    caps = get_capsules_9(q, base)
    for (i, j) in SELF_PAIRS:
        p1, p2, ri = caps[i]
        q1, q2, rj = caps[j]
        if seg_seg_dist(p1, p2, q1, q2) < ri + rj + SELF_MARGIN:
            return True
    return False


def any_self_collides(qs: Dict[str, np.ndarray]) -> bool:
    """True if ANY arm has a self-collision."""
    for name in ARM_NAMES:
        if self_collides_9(qs[name], ROBOT_BASES[name]):
            return True
    return False


def min_self_clearance_9(
    q: np.ndarray,
    base: np.ndarray,
) -> Tuple[float, str]:
    caps  = get_capsules_9(q, base)
    mn_d  = float('inf')
    mn_name = ''
    for (i, j) in SELF_PAIRS:
        p1, p2, ri = caps[i]
        q1, q2, rj = caps[j]
        d = seg_seg_dist(p1, p2, q1, q2) - ri - rj
        if d < mn_d:
            mn_d   = d
            mn_name = f'{SEG_NAMES[i]}↔{SEG_NAMES[j]}'
    return float(mn_d), mn_name


def all_self_clearances(
    qs: Dict[str, np.ndarray],
) -> Dict[str, Tuple[float, str]]:
    
    return {
        name: min_self_clearance_9(qs[name], ROBOT_BASES[name])
        for name in ARM_NAMES
    }