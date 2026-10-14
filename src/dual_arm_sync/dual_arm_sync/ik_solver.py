#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


L1 = 0.1525   
L2 = 0.620   
L3 = 0.559    
L4 = 0.121    
A  = 0.0345    


DH_PARAMS = np.array([
    # [ a,         alpha,       d,   theta_offset ]
    [0,    np.pi/2,  L1,  0],  
    [L2,   0,        0,   0],  
    [A,    np.pi/2,  0,   0],  
    [0,   -np.pi/2,  L3,  0],  
    [0,    np.pi/2,  0,   0], 
    [0,    0,        L4,  0],  
], dtype=float)


JOINT_LIMITS = np.array([
    [-2*np.pi,  2*np.pi],  
    [-1.6493,   1.6493 ],   
    [-2.7925,   2.7925 ],   
    [-2*np.pi,  2*np.pi],  
    [-2*np.pi,  2*np.pi],  
    [-2*np.pi,  2*np.pi], 
], dtype=float)

JOINT_VEL_MAX = np.array([2.09, 2.09, 3.14, 3.93, 3.93, 3.93]) 


IK_POS_TOL    = 0.002   
IK_ORI_WEIGHT = 0.3    


COLLISION_TOL = 0.002


LEX_TIE_FRAC = 0.05


class RobotBases:
    
    DSR01_BASE = np.array([0.0,  0.5, 0.0])
    DSR02_BASE = np.array([0.0, -0.5, 0.0])


class ArmRegistry:
    

    def __init__(self):
        self._arms: Dict[str, Dict] = {}

    def register(self, arm_id: str, base: np.ndarray, joints: np.ndarray):
        self._arms[arm_id] = {
            'base':   np.asarray(base,   dtype=float),
            'joints': np.asarray(joints, dtype=float),
        }

    def update_joints(self, arm_id: str, joints: np.ndarray):
        """Called on every joint-state callback to keep registry live."""
        if arm_id in self._arms:
            self._arms[arm_id]['joints'] = np.asarray(joints, dtype=float)

    def get_other_arms(self, arm_id: str) -> List[Dict]:
        """Return [{base, joints}, ...] for every arm except arm_id."""
        return [v for k, v in self._arms.items() if k != arm_id]

    def __len__(self):
        return len(self._arms)

    def __contains__(self, arm_id):
        return arm_id in self._arms


ARM_REGISTRY = ArmRegistry()


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr            ],
    ])


def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if abs(np.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw  = np.arctan2(-R[1, 2], R[1, 1]) if R[2, 0] < 0 else \
               np.arctan2( R[1, 2], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w)  ],
        [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w)  ],
        [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y)],
    ])


def slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    
    q0 = rotation_to_quaternion(R0)
    q1 = rotation_to_quaternion(R1)

    
    if np.dot(q0, q1) < 0:
        q1 = -q1

    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    theta = np.arccos(dot)

    if abs(theta) < 1e-8:
        qt = q0 + t * (q1 - q0)
    else:
        s0 = np.sin((1 - t) * theta) / np.sin(theta)
        s1 = np.sin(      t  * theta) / np.sin(theta)
        qt = s0 * q0 + s1 * q1

    return quaternion_to_matrix(qt / (np.linalg.norm(qt) + 1e-12))


def interpolate_orientations(R_start: np.ndarray,
                               R_end:   np.ndarray,
                               n:       int) -> List[np.ndarray]:
    
    s_vals = np.linspace(0.0, 1.0, n)
    return [slerp(R_start, R_end, float(s)) for s in s_vals]


def orientation_error(R_actual: np.ndarray, R_target: np.ndarray) -> float:
    
    R_err = R_target.T @ R_actual
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def get_target_rotation(target_local: np.ndarray) -> np.ndarray:
    
    if target_local[2] < 0.6:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    z_ax = target_local / (np.linalg.norm(target_local) + 1e-12)
    x_ax = np.cross([0, 0, 1], z_ax)
    xn   = np.linalg.norm(x_ax)
    x_ax = x_ax / xn if xn > 1e-6 else np.array([1., 0., 0.])
    return np.column_stack([x_ax, np.cross(z_ax, x_ax), z_ax])



LINK_SPHERES: List[Tuple[int, float, float]] = [
    (0, 0.00, 0.080), 
    (0, 0.08, 0.070),  
    (1, 0.10, 0.070), 
    (1, 0.30, 0.070), 
    (1, 0.50, 0.060),  
    (2, 0.00, 0.060), 
    (3, 0.10, 0.060), 
    (3, 0.30, 0.060),  
    (3, 0.50, 0.055),  
    (4, 0.00, 0.055),  
    (5, 0.00, 0.050),  
    (6, 0.00, 0.045),  
]
N_SPHERES = len(LINK_SPHERES)

SELF_COLLISION_PAIRS: List[Tuple[int, int]] = [
    (i, j)
    for i in range(N_SPHERES)
    for j in range(i + 2, N_SPHERES)
    if abs(LINK_SPHERES[i][0] - LINK_SPHERES[j][0]) > 2
]



def _std_dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0,    sa,      ca,     d   ],
        [0,    0,       0,      1   ],
    ])


def forward_kinematics(joints: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        a, alpha, d, theta_off = DH_PARAMS[i]
        theta = theta_off + joints[i]
        T = T @ _std_dh_matrix(a, alpha, d, theta)
        frames.append(T.copy())
    return T[:3, 3].copy(), T[:3, :3].copy(), frames



def get_sphere_centres(joints: np.ndarray,
                       base:   np.ndarray) -> np.ndarray:
    """All N_SPHERES sphere centres in WORLD frame.  Shape (N_SPHERES, 3)."""
    _, _, frames = forward_kinematics(joints)
    centres = np.zeros((N_SPHERES, 3))
    for k, (fi, zo, _) in enumerate(LINK_SPHERES):
        T = frames[fi]
        centres[k] = T[:3, 3] + zo * T[:3, 2] + base
    return centres

def self_collision_score(joints: np.ndarray) -> float:
    """Summed penetration depth for non-adjacent link sphere pairs (local frame)."""
    _, _, frames = forward_kinematics(joints)
    centres = np.zeros((N_SPHERES, 3))
    for k, (fi, zo, _) in enumerate(LINK_SPHERES):
        T = frames[fi]
        centres[k] = T[:3, 3] + zo * T[:3, 2]
    total = 0.0
    for i, j in SELF_COLLISION_PAIRS:
        d = np.linalg.norm(centres[i] - centres[j])
        total += max(0.0, LINK_SPHERES[i][2] + LINK_SPHERES[j][2] - d)
    return total
    return total


def inter_arm_collision_score(joints:      np.ndarray,
                               base:       np.ndarray,
                               other_arms: List[Dict]) -> float:
    """Summed penetration depth vs ALL other arms (world frame)."""
    if not other_arms:
        return 0.0
    my_c = get_sphere_centres(joints, base)
    total = 0.0
    for arm in other_arms:
        oc = get_sphere_centres(arm['joints'], arm['base'])
        for i in range(N_SPHERES):
            for j in range(N_SPHERES):
                d = np.linalg.norm(my_c[i] - oc[j])
                total += max(0.0, LINK_SPHERES[i][2] + LINK_SPHERES[j][2] - d)
    return total


def total_collision_score(joints:      np.ndarray,
                           base:       np.ndarray,
                           other_arms: List[Dict]) -> float:
    return self_collision_score(joints) + \
           inter_arm_collision_score(joints, base, other_arms)


def compute_jacobian(joints: np.ndarray) -> np.ndarray:
    """Geometric Jacobian  J ∈ ℝ^{6×6}."""
    _, _, frames = forward_kinematics(joints)
    p_ee = frames[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J


def manipulability(joints: np.ndarray) -> float:
    """Yoshikawa index  w = √det(J·Jᵀ).   0 at singularity."""
    try:
        J = compute_jacobian(joints)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
    except Exception:
        return 0.0

def joint_limit_margin(joints: np.ndarray) -> float:
    """
    Minimum normalised distance to any joint boundary.
    Range [0, 0.5].  0 = at limit,  0.5 = at midpoint.
    """
    margins = []
    for j in range(6):
        lo, hi = JOINT_LIMITS[j]
        margins.append(min(joints[j] - lo, hi - joints[j]) / (hi - lo))
    return float(min(margins))

def _normalize_joints_to_current(solution:       np.ndarray,
                                   current_joints: np.ndarray) -> np.ndarray:
    
    sol = solution.copy()
    for j in range(6):
        lo, hi = JOINT_LIMITS[j]
        base   = sol[j]
        best   = base
        bdist  = abs(base - current_joints[j])
        for k in range(-3, 4):          
            cand = base + k * 2.0 * np.pi
            if lo - 1e-9 <= cand <= hi + 1e-9:
                cand   = float(np.clip(cand, lo, hi))
                d      = abs(cand - current_joints[j])
                closer = d < bdist - 1e-9
                
                tiebreak = (abs(d - bdist) < 1e-9) and (abs(cand) < abs(best) - 1e-9)
                if closer or tiebreak:
                    bdist = d
                    best  = cand
        sol[j] = best
    return sol




def _tier(values: np.ndarray, higher_better: bool) -> np.ndarray:
    rng = values.max() - values.min()
    if rng < 1e-12:
        return np.zeros(len(values), dtype=int)
    threshold = LEX_TIE_FRAC * rng
    best = values.max() if higher_better else values.min()
    in_band = (values >= best - threshold) if higher_better \
              else (values <= best + threshold)
    return np.where(in_band, 0, 1).astype(int)


def select_optimal_solution(solutions:      List[np.ndarray],
                             current_joints: np.ndarray,
                             arm_id:         Optional[str]        = None,
                             base:           Optional[np.ndarray] = None,
                             other_arms:     Optional[List[Dict]] = None,
                             verbose:        bool = True
                             ) -> Optional[Tuple[np.ndarray, Dict]]:
    
    if not solutions:
        return None

    if other_arms is None:
        other_arms = ARM_REGISTRY.get_other_arms(arm_id) if arm_id else []
    if base is None:
        base = ARM_REGISTRY._arms[arm_id]['base'] \
               if (arm_id and arm_id in ARM_REGISTRY) else np.zeros(3)

    
    norm_sols = [_normalize_joints_to_current(s, current_joints) for s in solutions]

    N     = len(norm_sols)
    col_s = np.array([total_collision_score(s, base, other_arms) for s in norm_sols])
    man_s = np.array([manipulability(s)                          for s in norm_sols])
    lim_s = np.array([joint_limit_margin(s)                      for s in norm_sols])
    dis_s = np.array([np.linalg.norm(s - current_joints)         for s in norm_sols])

    if verbose:
        print(f'\n  [Lex]  candidates={N}  other_arms={len(other_arms)}')
        print(f"  {'':>4} {'Collision':>10} {'Manip':>10} {'Margin':>8} {'Disp°':>8}")
        for k in range(N):
            tag = '✓' if col_s[k] <= COLLISION_TOL else '✗'
            print(f"  [{k}]{tag} {col_s[k]:>10.4f} {man_s[k]:>10.4f} "
                  f"{lim_s[k]:>8.3f} {np.degrees(dis_s[k]):>8.1f}")

    cf_mask = col_s <= COLLISION_TOL
    active  = np.where(cf_mask)[0] if cf_mask.any() else np.arange(N)
    if verbose and not cf_mask.any():
        print('  ⚠  All in collision — selecting least-bad')

    t1    = _tier(col_s[active], higher_better=False)
    tied1 = active[t1 == 0]
    t2    = _tier(man_s[tied1], higher_better=True)
    tied2 = tied1[t2 == 0]
    t3    = _tier(lim_s[tied2], higher_better=True)
    tied3 = tied2[t3 == 0]
    winner = int(tied3[np.argmin(dis_s[tied3])])

    if verbose:
        print(f'  P1→{tied1.tolist()}  P2→{tied2.tolist()}  '
              f'P3→{tied3.tolist()}  winner=[{winner}]')

    sol  = norm_sols[winner]   
    info = {
        'collision_score':  float(col_s[winner]),
        'self_collision':   float(self_collision_score(sol)),
        'inter_collision':  float(inter_arm_collision_score(sol, base, other_arms)),
        'manipulability':   float(man_s[winner]),
        'limit_margin':     float(lim_s[winner]),
        'displacement':     float(dis_s[winner]),
        'collision_free':   bool(col_s[winner] <= COLLISION_TOL),
        'num_solutions':    N,
        'num_free':         int(cf_mask.sum()),
        'n_other_arms':     len(other_arms),
        'cost':             float(col_s[winner]),
    }
    return sol, info


def select_chained_solution(solutions:   List[np.ndarray],
                             prev_joints: np.ndarray,
                             arm_id:      Optional[str]        = None,
                             base:        Optional[np.ndarray] = None,
                             other_arms:  Optional[List[Dict]] = None
                             ) -> Optional[np.ndarray]:
    
    result = select_optimal_solution(
        solutions, prev_joints,
        arm_id=arm_id, base=base, other_arms=other_arms,
        verbose=False,
    )
    return result[0] if result else None

def _ik_cost(joints:   np.ndarray,
             tgt_pos:  np.ndarray,
             tgt_rot:  np.ndarray,
             w_orient: float) -> float:
    
    try:
        pos, rot, _ = forward_kinematics(joints)
    except Exception:
        return 1e6
    pos_err = float(np.linalg.norm(pos - tgt_pos))
    if tgt_rot is not None:
        ori_err = orientation_error(rot, tgt_rot)
        return pos_err + w_orient * ori_err
    return pos_err


def solve_ik_numerical(target_local:  np.ndarray,
                        target_orient: Optional[np.ndarray] = None,
                        initial_guess: Optional[np.ndarray] = None,
                        n_restarts:    int   = 8,
                        pos_tol:       float = IK_POS_TOL,
                        orient_weight: float = IK_ORI_WEIGHT
                        ) -> List[np.ndarray]:
    
    if target_orient is None:
        target_orient = get_target_rotation(target_local)

    bounds    = [(JOINT_LIMITS[j, 0], JOINT_LIMITS[j, 1]) for j in range(6)]
    solutions = []

    for i in range(n_restarts):
        if i == 0 and initial_guess is not None:
            x0 = initial_guess.copy()
        else:
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

        res = minimize(
            _ik_cost, x0,
            args=(target_local, target_orient, orient_weight),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-8},
        )
        if res.fun >= pos_tol + orient_weight * np.pi:
            continue
        try:
            fk_pos, fk_rot, _ = forward_kinematics(res.x)
        except Exception:
            continue
        if np.linalg.norm(fk_pos - target_local) > pos_tol:
            continue
        
        if not any(np.linalg.norm(res.x - s) < 0.05 for s in solutions):
            solutions.append(res.x.copy())

    return solutions

def compute_cost(joints: np.ndarray, current_joints: np.ndarray) -> float:
    
    col  = self_collision_score(joints)
    man  = manipulability(joints)
    disp = float(np.linalg.norm(joints - current_joints))
    return col * 10.0 + max(0.0, 0.1 - man) * 2.0 + disp