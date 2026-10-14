#!/usr/bin/env python3

import json, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

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

POS_LIM = np.array([
    [-2*_PI,  2*_PI ], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
NDOF    = 6

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}
ROBOT_NAMES = ['dsr01', 'dsr02']

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12

IK_TOL_POS  = 0.010   
IK_TOL_ROT  = 0.05   
IK_MAX_ITER = 800
IK_FTOL     = 1e-10
IK_UNIQ     = 0.12

W_POS = 1.0
W_ROT = 0.15

W_CLEAR = 3.0  
W_TIME  = 2.0   
W_LIM   = 1.0  
W_MANIP = 0.4   

T_REF = 5.0   

def fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
    return T[:3, 3].copy(), T


def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o


def rot_err(R_got: np.ndarray, R_tgt: np.ndarray) -> float:
    R = R_tgt @ R_got.T
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def manipulability(q: np.ndarray) -> float:
    eps = 1e-5; J = np.zeros((3, NDOF)); p0 = fk(q)[0]
    for i in range(NDOF):
        dq = q.copy(); dq[i] += eps; J[:, i] = (fk(dq)[0] - p0) / eps
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def pair_min_dist(qi, bi, qj, bj) -> float:
    """Minimum link-origin distance — vectorised broadcasting."""
    oi = link_origins(qi, bi)  # (NDOF,3)
    oj = link_origins(qj, bj)  # (NDOF,3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]  # (NDOF,NDOF,3)
    return float(np.min(np.linalg.norm(diff, axis=2)))

def quat_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

def min_motion_time(q: np.ndarray, start_q: np.ndarray) -> float:
    return float(np.max(np.abs(q - start_q) / VEL_LIM))


def sq_vel_cost(q: np.ndarray, start_q: np.ndarray) -> float:
    dq = q - start_q
    return float(np.sum((dq / VEL_LIM) ** 2))

def ik_seeds(current: np.ndarray, tgt: np.ndarray) -> List[np.ndarray]:
    px, py, pz = tgt
    cl = lambda q: np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])

    seeds = [cl(current.copy())]

    for j in range(NDOF):
        for d in (0.3, -0.3, 0.6, -0.6):
            s = current.copy(); s[j] += d
            seeds.append(cl(s))

    t1 = float(np.arctan2(py, px))
    rh = float(np.hypot(px, py))
    re = float(np.sqrt(max(rh**2 - A**2, 0.0)))
    h  = float(pz - L1)
    c3 = float(np.clip((re**2 + h**2 - L2**2 - L3**2) / (2*L2*L3), -1, 1))
    for sgn in (1., -1.):
        th3 = sgn * float(np.arccos(c3))
        q3  = th3 - _PI_2
        th2 = float(np.arctan2(h, re)) - float(np.arctan2(L3*np.sin(th3), L2+L3*np.cos(th3)))
        q2  = th2 + _PI_2
        for q5 in (0., _PI_2, -_PI_2):
            for t in (t1, t1 + _PI_2, t1 - _PI_2):
                seeds.append(cl(np.array([t, q2, q3, 0., q5, 0.])))

    unique: List[np.ndarray] = []
    for s in seeds:
        if all(np.linalg.norm(s - u) > 0.05 for u in unique):
            unique.append(s)
    return unique

def solve_ik(target_local: np.ndarray,
             target_rot  : np.ndarray,
             current     : np.ndarray) -> List[np.ndarray]:
    bds = [(POS_LIM[i, 0], POS_LIM[i, 1]) for i in range(NDOF)]

    def obj(q):
        p, T = fk(q)
        return (W_POS * float(np.sum((p - target_local)**2))
              + W_ROT * float(rot_err(T[:3,:3], target_rot)**2))

    solutions: List[np.ndarray] = []
    for seed in ik_seeds(current, target_local):
        res = minimize(obj, seed, method='SLSQP', bounds=bds,
                       options={'maxiter': IK_MAX_ITER, 'ftol': IK_FTOL})
        if not res.success: continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        p, T = fk(q)
        if np.linalg.norm(p - target_local) >= IK_TOL_POS: continue
        if rot_err(T[:3,:3], target_rot)    >= IK_TOL_ROT:  continue
        if all(np.linalg.norm(q - s) > IK_UNIQ for s in solutions):
            solutions.append(q)
    return solutions

def score(q       : np.ndarray,
          start_q : np.ndarray,
          base    : np.ndarray,
          others  : List[Tuple[np.ndarray, np.ndarray]]) -> float:
    
    min_d = min(pair_min_dist(q, base, oq, ob) for oq, ob in others) if others else 1.0
    clear = float(np.tanh(min_d / 0.25))

    time_score = float(np.exp(-sq_vel_cost(q, start_q) / T_REF**2))

    mid = (POS_LIM[:, 0] + POS_LIM[:, 1]) / 2.0
    rng = POS_LIM[:, 1] - POS_LIM[:, 0]
    lim = float(np.mean(1.0 - 2.0 * np.abs(q - mid) / rng))

    manip = float(np.tanh(manipulability(q) / 0.05))

    return W_CLEAR * clear + W_TIME * time_score + W_LIM * lim + W_MANIP * manip

def find_best_configs(arms_data: Dict) -> Optional[Dict[str, np.ndarray]]:
    
    arm_names = sorted(arms_data.keys())
    all_sols:  Dict[str, List[np.ndarray]] = {}
    best:      Dict[str, np.ndarray]       = {}

    for name in arm_names:
        d     = arms_data[name]
        start = d['start_q']
        tloc  = d['target_local']
        trot  = d['target_rot']

        print(f'  [{name}] IK solving ...')
        sols = solve_ik(tloc, trot, start)

        if not sols:
            print(f'  [{name}] SE(3) failed — position-only fallback ...')
            bds = [(POS_LIM[i,0], POS_LIM[i,1]) for i in range(NDOF)]
            def pos_obj(q, tl=tloc):
                p, _ = fk(q); return float(np.sum((p - tl)**2))
            for seed in ik_seeds(start, tloc):
                res = minimize(pos_obj, seed, method='SLSQP', bounds=bds,
                               options={'maxiter': IK_MAX_ITER, 'ftol': IK_FTOL})
                q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
                if np.linalg.norm(fk(q)[0] - tloc) < IK_TOL_POS:
                    sols.append(q); break

        if not sols:
            print(f'  [{name}] ❌  No IK solution found')
            return None

        all_sols[name] = sols

        best[name] = min(sols, key=lambda q, s=start: min_motion_time(q, s))
        t_tmp = min_motion_time(best[name], start)
        print(f'  [{name}] {len(sols)} sol(s)  temp-best T≈{t_tmp:.2f}s  '
              f'sq_cost={sq_vel_cost(best[name], start):.4f}')
    for name in arm_names:
        base    = arms_data[name]['base']
        start_q = arms_data[name]['start_q']  
        others  = [(best[n], arms_data[n]['base']) for n in arm_names if n != name]

        scores     = [score(q, start_q, base, others) for q in all_sols[name]]
        best[name] = all_sols[name][int(np.argmax(scores))]

        d_clear = (pair_min_dist(
            best[name], base,
            best[[n for n in arm_names if n != name][0]],
            arms_data[[n for n in arm_names if n != name][0]]['base'])
            if len(arm_names) > 1 else 0.0)
        t_fin = min_motion_time(best[name], start_q)
        print(f'  [{name}] score={max(scores):.4f}  '
              f'clearance={d_clear*100:.1f}cm  T≈{t_fin:.2f}s')

    return best

if _ROS_OK:
    class Step1Node(Node):
        def __init__(self):
            super().__init__('step_1_ik')
            self._arm = {n: {'q': np.zeros(NDOF), 'ready': False}
                         for n in ROBOT_NAMES}
            for name in ROBOT_NAMES:
                for topic in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._arm[name]['q'] = q.astype(float)
            if not self._arm[name]['ready']:
                self._arm[name]['ready'] = True
                self.get_logger().info(f'{name} ready')

        def both_ready(self): return all(v['ready'] for v in self._arm.values())
        def joints(self, n):  return self._arm[n]['q'].copy()

        def wait(self, t=20.0):
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.both_ready(): return True
            return False

def _prompt_pose(label: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f'\n  [{label}] target pose:')
    x = float(input('    pos X (m): '))
    y = float(input('    pos Y (m): '))
    z = float(input('    pos Z (m): '))
    raw = input('    quaternion [w,x,y,z] (Enter=identity): ').strip()
    if raw:
        vals = [float(v) for v in raw.replace(',', ' ').split()]
        quat = quat_norm(np.array(vals[:4]))
    else:
        quat = np.array([1., 0., 0., 0.])
    return np.array([x, y, z]), quat


def main(args=None):
    print('\n' + '=' * 68)
    print('  STEP 1  —  IK: Position + Orientation → Target Joint Configs')
    print('=' * 68)

    if _ROS_OK:
        rclpy.init(args=args)
        node = Step1Node()
        print('\n  Waiting for joint states ...')
        if not node.wait(20.0):
            print('  ❌  Timeout'); node.destroy_node(); rclpy.shutdown(); return
        print('  ✅  Both arms ready')
        get_q = node.joints
        spin  = lambda: rclpy.spin_once(node, timeout_sec=0.05)
    else:
        print('  ⚠   No ROS2 — using zero start configs')
        get_q = lambda _: np.zeros(NDOF)
        spin  = lambda: None

    try:
        while True:
            print('\n' + '─' * 68)
            poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for name in ROBOT_NAMES:
                poses[name] = _prompt_pose(name.upper())
            dur_raw  = input('\n  Duration [s] (Enter=10): ').strip()
            duration = float(dur_raw) if dur_raw else 10.0
            spin()

            arms_data: Dict = {}
            for name in ROBOT_NAMES:
                base     = ROBOT_BASES[name]
                start_q  = get_q(name)
                tw, quat = poses[name]
                arms_data[name] = {
                    'base'        : base,
                    'start_q'     : start_q,
                    'target_world': tw,
                    'target_local': tw - base,
                    'target_rot'  : quat_to_rot(quat),
                    'target_quat' : quat,
                }

            print('\n  Current joint configs (start):')
            for name in ROBOT_NAMES:
                deg = [round(float(np.degrees(v)), 2) for v in arms_data[name]['start_q']]
                print(f'    [{name}] {deg} deg')

            print('\n  Solving IK ...')
            best = find_best_configs(arms_data)
            if best is None:
                print('  ❌  IK failed for one or more arms — adjust targets')
                continue

            if len(ROBOT_NAMES) >= 2:
                d = pair_min_dist(best[ROBOT_NAMES[0]], ROBOT_BASES[ROBOT_NAMES[0]],
                                   best[ROBOT_NAMES[1]], ROBOT_BASES[ROBOT_NAMES[1]])
                icon = '✅' if d > SAFETY_MARGIN else '⚠ '
                print(f'\n  {icon}  Inter-arm clearance at targets: {d*100:.1f}cm')

            out = {'duration': duration}
            for name in ROBOT_NAMES:
                base     = arms_data[name]['base']
                start_q  = arms_data[name]['start_q']
                target_q = best[name]
                ps, _    = fk(start_q);  ps_w = (ps + base).tolist()
                pt, Tt   = fk(target_q); pt_w = (pt + base).tolist()
                pos_err  = float(np.linalg.norm(pt - arms_data[name]['target_local']) * 1000)
                rot_e    = float(np.degrees(rot_err(Tt[:3,:3], arms_data[name]['target_rot'])))
                t_move   = min_motion_time(target_q, start_q)
                cost     = sq_vel_cost(target_q, start_q)

                out[name] = {
                    'robot_name'            : name,
                    'base'                  : base.tolist(),
                    'start_joints'          : start_q.tolist(),
                    'target_joints'         : target_q.tolist(),
                    'start_joints_deg'      : np.degrees(start_q).tolist(),
                    'target_joints_deg'     : np.degrees(target_q).tolist(),
                    'start_pos_world'       : ps_w,
                    'target_pos_world'      : pt_w,
                    'target_quaternion'     : arms_data[name]['target_quat'].tolist(),
                    'position_error_mm'     : round(pos_err, 3),
                    'orientation_error_deg' : round(rot_e, 3),
                    'min_motion_time_s'     : round(t_move, 3),
                    'sq_vel_cost'           : round(cost, 5),
                }
                print(f'  [{name}] pos={pos_err:.2f}mm  rot={rot_e:.2f}°  '
                      f'T≈{t_move:.2f}s  sq_cost={cost:.4f}')

            with open('ik_solutions.json', 'w') as fh:
                json.dump(out, fh, indent=2)
            print('\n  ✅  Saved: ik_solutions.json  →  run step_2.py')

            if input('\n  Again? (y/n): ').strip().lower() != 'y':
                break

    except KeyboardInterrupt:
        print('\n  Interrupted.')

    if _ROS_OK:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()