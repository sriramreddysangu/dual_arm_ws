#!/usr/bin/env python3

import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


try:
    from dual_arm_sync.ik_solver import (
        forward_kinematics,  
        RobotBases,         
        JOINT_LIMITS,        
        JOINT_VEL_MAX,       
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        forward_kinematics, RobotBases, JOINT_LIMITS, JOINT_VEL_MAX,
    )


class Config:
    
    MIN_SAFE_DIST  = 0.15   
    WARNING_DIST   = 0.20  
    REP_THRESH     = 0.25   
    LINK_RADII     = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])

    
    BASE_COUPLING  = 2.0    
    MAX_COUPLING   = 12.0   
    REPULSION_STR  = 80.0   
    EMERGENCY_STR  = 200.0 
    MAX_PHASE_VEL  = 2.0   
    ADAPT_RATE     = 0.5   
    RECOVERY_RATE  = 0.1   
    DT             = 0.01   


    MAX_ITER         = 10   
    CP_ADD_PER_ITER  = 2    
    PERTURB_SCALE_0  = 0.12 
    ANNEAL_RATE      = 0.80 
    RANDOM_TRIES     = 30  
    MAX_SEGMENTS     = 32   



def all_link_positions(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    
    _, _, frames = forward_kinematics(q)   # Standard DH — matches ik_solver.py
    return np.array([frames[i][:3, 3] + base for i in range(1, 7)])


def min_link_distance(q1: np.ndarray, q2: np.ndarray,
                       base1: np.ndarray, base2: np.ndarray) -> float:
    
    pts1 = all_link_positions(q1, base1)
    pts2 = all_link_positions(q2, base2)
    d = np.inf
    for a in pts1:
        for b in pts2:
            d = min(d, float(np.linalg.norm(a - b)))
    return d


def rebuild_bspline(waypoints: np.ndarray,
                    t_norm:    np.ndarray,
                    T:         float,
                    n_eval:    int = 1000) -> Dict:
    
    pos = np.zeros((n_eval, 6))
    vel = np.zeros((n_eval, 6))
    acc = np.zeros((n_eval, 6))
    t_ev = np.linspace(0.0, 1.0, n_eval)

    for j in range(6):
        k   = min(3, len(waypoints) - 1)
        tck = splrep(t_norm, waypoints[:, j], k=k, s=0)
        pos[:, j] = splev(t_ev, tck, der=0)
        vel[:, j] = splev(t_ev, tck, der=1) / T
        acc[:, j] = splev(t_ev, tck, der=2) / (T ** 2)

    return {
        'time':            t_ev * T,
        'positions':       pos,
        'velocities':      vel,
        'accelerations':   acc,
        'joint_waypoints': waypoints.copy(),
        'waypoint_times':  t_norm * T,
        'num_samples':     n_eval,
    }


def interpolate_at(traj: Dict, t: float) -> np.ndarray:
    
    tv  = traj['time']
    pos = traj['positions']
    if t <= tv[0]:   return pos[0].copy()
    if t >= tv[-1]:  return pos[-1].copy()
    idx = np.searchsorted(tv, t)
    lo, hi = idx - 1, idx
    alpha  = (t - tv[lo]) / (tv[hi] - tv[lo] + 1e-12)
    return (1 - alpha) * pos[lo] + alpha * pos[hi]



class KuramotoSync:
    

    def __init__(self, traj1: Dict, traj2: Dict,
                 base1: np.ndarray, base2: np.ndarray):
        self.traj1  = traj1
        self.traj2  = traj2
        self.base1  = base1
        self.base2  = base2
        self.T      = max(float(traj1['time'][-1]), float(traj2['time'][-1]))
        self.omega0 = 1.0 / self.T

    def _q(self, phi: float, traj: Dict) -> np.ndarray:
        """Sample joint config at normalised phase φ ∈ [0,1]."""
        pos = traj['positions']
        n   = len(pos)
        f   = float(np.clip(phi, 0.0, 1.0))
        lo  = int(f * (n - 1))
        hi  = min(lo + 1, n - 1)
        a   = f * (n - 1) - lo
        return (1 - a) * pos[lo] + a * pos[hi]

    def _deriv(self, state: np.ndarray, t: float) -> np.ndarray:
        phi1, phi2, w1, w2 = state

        q1 = self._q(phi1, self.traj1)
        q2 = self._q(phi2, self.traj2)

        dist   = min_link_distance(q1, q2, self.base1, self.base2)
        prox   = float(np.clip(1.0 - dist / Config.REP_THRESH,   0, 1))
        danger = float(np.clip(1.0 - dist / Config.MIN_SAFE_DIST, 0, 1))

        
        d_phi = phi1 - phi2
        leader = 0
        if   d_phi >  0.05: leader =  1  
        elif d_phi < -0.05: leader = -1  

       
        K  = min(Config.BASE_COUPLING * (1.0 + 3.0 * prox), Config.MAX_COUPLING)
        c1 = K * np.sin(phi2 - phi1)   
        c2 = K * np.sin(phi1 - phi2)

      
        rep1 = rep2 = 0.0
        if dist < Config.REP_THRESH:
            mag = Config.REPULSION_STR * prox ** 2
            if   leader ==  1: rep1 =  mag;       rep2 = -mag * 0.3
            elif leader == -1: rep2 =  mag;       rep1 = -mag * 0.3
            else:              rep1 = rep2 = mag * 0.7

       
        if dist < Config.MIN_SAFE_DIST:
            emerg = Config.EMERGENCY_STR * danger ** 3
            if   leader ==  1: rep1 += emerg * 2.0; rep2 += emerg * 0.5
            elif leader == -1: rep2 += emerg * 2.0; rep1 += emerg * 0.5
            else:              rep1 += emerg;        rep2 += emerg

        dphi1 = float(np.clip(w1 + c1 - rep1,
                               -Config.MAX_PHASE_VEL, Config.MAX_PHASE_VEL))
        dphi2 = float(np.clip(w2 + c2 - rep2,
                               -Config.MAX_PHASE_VEL, Config.MAX_PHASE_VEL))

   
        dw1 = dw2 = 0.0
        if dist < Config.REP_THRESH:
            if   leader ==  1: dw1 = -Config.ADAPT_RATE * prox * w1
            elif leader == -1: dw2 = -Config.ADAPT_RATE * prox * w2
        else:
            dw1 = Config.RECOVERY_RATE * (w2 - w1)
            dw2 = Config.RECOVERY_RATE * (w1 - w2)

        return np.array([dphi1, dphi2, dw1, dw2])

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        t_vec  = np.arange(0.0, self.T + Config.DT, Config.DT)
        state0 = np.array([0.0, 0.0, self.omega0, self.omega0])
        sol    = odeint(self._deriv, state0, t_vec, rtol=1e-4, atol=1e-6)
        phi1   = np.clip(sol[:, 0], 0.0, 1.0)
        phi2   = np.clip(sol[:, 1], 0.0, 1.0)
        return phi1, phi2, t_vec

    def reparameterise(self,
                        phi1:  np.ndarray,
                        phi2:  np.ndarray,
                        t_vec: np.ndarray
                        ) -> Tuple[Dict, Dict]:
        
        def _resamp(phi: np.ndarray, traj: Dict) -> Dict:
            pos = traj['positions']
            n   = len(pos)
            new_pos = np.zeros((len(phi), 6))
            for k, p in enumerate(phi):
                f  = float(np.clip(p, 0.0, 1.0))
                lo = int(f * (n - 1))
                hi = min(lo + 1, n - 1)
                a  = f * (n - 1) - lo
                new_pos[k] = (1 - a) * pos[lo] + a * pos[hi]
            nv = np.gradient(new_pos, t_vec, axis=0)
            na = np.gradient(nv,      t_vec, axis=0)
            return {
                'time':            t_vec.copy(),
                'positions':       new_pos,
                'velocities':      nv,
                'accelerations':   na,
                'joint_waypoints': traj.get('joint_waypoints',
                                            new_pos[[0, -1]]),
                'waypoint_times':  traj.get('waypoint_times',
                                            np.array([0.0, t_vec[-1]])),
                'num_samples':     len(t_vec),
            }

        return _resamp(phi1, self.traj1), _resamp(phi2, self.traj2)

def scan_collision_indices(traj1: Dict, traj2: Dict,
                            base1: np.ndarray, base2: np.ndarray,
                            n_check: int = 300) -> List[int]:
    
    n    = min(len(traj1['positions']), len(traj2['positions']))
    step = max(1, n // n_check)
    bad  = []
    for idx in range(0, n, step):
        d = min_link_distance(traj1['positions'][idx],
                               traj2['positions'][idx],
                               base1, base2)
        if d < Config.MIN_SAFE_DIST + 0.01:   # 1 cm buffer
            bad.append(idx)
    return bad


def is_collision_free(traj1: Dict, traj2: Dict,
                       base1: np.ndarray, base2: np.ndarray,
                       n_check: int = 300) -> Tuple[bool, float]:
   
    n     = min(len(traj1['positions']), len(traj2['positions']))
    step  = max(1, n // n_check)
    min_d = np.inf
    for idx in range(0, n, step):
        d     = min_link_distance(traj1['positions'][idx],
                                   traj2['positions'][idx],
                                   base1, base2)
        min_d = min(min_d, d)
    return (min_d >= Config.MIN_SAFE_DIST), float(min_d)


class ControlPointRefinement:
    

    @staticmethod
    def _cluster_windows(bad_indices: List[int],
                          n_pts:       int,
                          padding:     float = 0.08
                          ) -> List[Tuple[float, float]]:
        """
        Group consecutive bad indices into (lo, hi) normalised time windows.
        """
        if not bad_indices:
            return []
        clusters, current = [], [bad_indices[0]]
        for idx in bad_indices[1:]:
            gap = max(3, n_pts // 50)
            if idx - current[-1] <= gap:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)

        windows = []
        for cl in clusters:
            lo = max(0.0, (min(cl) / n_pts) - padding)
            hi = min(1.0, (max(cl) / n_pts) + padding)
            windows.append((lo, hi))
        return windows

    @staticmethod
    def refine(traj:       Dict,
               other_traj: Dict,
               traj_base:  np.ndarray,
               other_base: np.ndarray,
               bad_indices: List[int],
               iteration:   int) -> Dict:
        
        wps    = traj['joint_waypoints'].copy()   
        t_way  = traj['waypoint_times'].copy()   
        T      = float(traj['time'][-1])
        t_norm = t_way / T                       
        n_pts  = len(traj['positions'])

        windows = ControlPointRefinement._cluster_windows(bad_indices, n_pts)
        if not windows:
            return traj

        
        for (lo, hi) in windows:
            n_new = Config.CP_ADD_PER_ITER + iteration // 2
            new_t = np.linspace(lo, hi, n_new + 2)[1:-1]  
            for nt in new_t:
                nearest = int(np.argmin(np.abs(t_norm - nt)))
                wps     = np.vstack([wps,    wps[nearest].copy()[None, :]])
                t_norm  = np.append(t_norm,  nt)

        
        order  = np.argsort(t_norm)
        t_norm = t_norm[order]
        wps    = wps[order]

        
        _, unique = np.unique(np.round(t_norm, 6), return_index=True)
        t_norm = t_norm[unique]
        wps    = wps[unique]

        
        scale    = Config.PERTURB_SCALE_0 * (Config.ANNEAL_RATE ** iteration)
        best_wps = wps.copy()
        best_bad = len(bad_indices)

       
        j_lo = JOINT_LIMITS[:, 0]   
        j_hi = JOINT_LIMITS[:, 1]   

        for _ in range(Config.RANDOM_TRIES):
            cand = wps.copy()
            for k, tn in enumerate(t_norm):
                in_window  = any(lo <= tn <= hi for (lo, hi) in windows)
                is_endpoint = (tn < 0.01 or tn > 0.99)
                if in_window and not is_endpoint:
                    cand[k] = np.clip(cand[k] + np.random.randn(6) * scale,
                                       j_lo, j_hi)

            cand_traj = rebuild_bspline(cand, t_norm, T)

            
            step  = max(1, len(cand_traj['positions']) // 150)
            n_bad = 0
            for idx in range(0, len(cand_traj['positions']), step):
                oi = min(idx, len(other_traj['positions']) - 1)
                d  = min_link_distance(cand_traj['positions'][idx],
                                        other_traj['positions'][oi],
                                        traj_base, other_base)
                if d < Config.MIN_SAFE_DIST:
                    n_bad += 1

            if n_bad < best_bad:
                best_bad = n_bad
                best_wps = cand.copy()

        return rebuild_bspline(best_wps, t_norm, T)



def synchronize(traj1_in: Dict, traj2_in: Dict,
                base1:    np.ndarray, base2: np.ndarray
                ) -> Tuple[Optional[Dict], Optional[Dict], Dict]:
    
    print("\n" + "=" * 80)
    print("KURAMOTO SYNCHRONIZATION  —  Iterative Refinement  (Paper §IV)")
    print(f"Max iterations   : {Config.MAX_ITER}")
    print(f"Min safe dist    : {Config.MIN_SAFE_DIST * 100:.0f} cm")
    print(f"Repulsion onset  : {Config.REP_THRESH * 100:.0f} cm")
    print(f"Perturb scale_0  : {Config.PERTURB_SCALE_0:.3f} rad  "
          f"(×{Config.ANNEAL_RATE} per iter)")
    print("=" * 80)

    traj1 = traj1_in
    traj2 = traj2_in

    
    free, min_d = is_collision_free(traj1, traj2, base1, base2)
    status = '✅ collision-free' if free else '❌ collision detected'
    print(f"\nInitial : {status}  |  min clearance = {min_d * 100:.1f} cm")

    if free:
        return traj1, traj2, _make_report(True, 0, min_d,
                                           "Already collision-free — no sync needed")

    
    for iteration in range(1, Config.MAX_ITER + 1):
        print(f"\n{'─' * 80}")
        print(f"ITERATION {iteration} / {Config.MAX_ITER}")
        print(f"{'─' * 80}")

        
        print("  [①] Kuramoto phase coupling...")
        kura = KuramotoSync(traj1, traj2, base1, base2)
        phi1, phi2, t_vec = kura.run()
        t1_new, t2_new    = kura.reparameterise(phi1, phi2, t_vec)
        print(f"      Integration: {len(t_vec)} steps  T = {t_vec[-1]:.2f}s")

        
        free, min_d = is_collision_free(t1_new, t2_new, base1, base2)
        print(f"  [②] After Kuramoto : {'✅ clear' if free else '❌ colliding'}"
              f"  min = {min_d * 100:.1f} cm")

        if free:
            report = _make_report(True, iteration, min_d,
                                   f"Resolved by Kuramoto at iteration {iteration}")
            print(f"\n✅  Collision-free after {iteration} iteration(s)  "
                  f"(Kuramoto sufficient)  min_clr={min_d * 100:.1f} cm")
            return t1_new, t2_new, report

        
        bad = scan_collision_indices(t1_new, t2_new, base1, base2)
        print(f"  [③] Collision indices: {len(bad)}"
              f"  (paper §IV-B: insert control points in windows)")

        
        phi1_mean = float(np.mean(phi1))
        phi2_mean = float(np.mean(phi2))
        lead = 1 if phi1_mean >= phi2_mean else 2
        scale_now = Config.PERTURB_SCALE_0 * (Config.ANNEAL_RATE ** (iteration - 1))
        print(f"  [④] Perturbing arm dsr0{lead}  "
              f"(leading: φ_mean = {phi1_mean:.3f} vs {phi2_mean:.3f})"
              f"  σ = {scale_now:.4f} rad")

        if lead == 1:
            t1_new = ControlPointRefinement.refine(
                t1_new, t2_new, base1, base2, bad, iteration)
        else:
            t2_new = ControlPointRefinement.refine(
                t2_new, t1_new, base2, base1, bad, iteration)

        n1 = len(t1_new.get('joint_waypoints', []))
        n2 = len(t2_new.get('joint_waypoints', []))
        print(f"      Waypoints: dsr01={n1}  dsr02={n2}")

        
        free, min_d = is_collision_free(t1_new, t2_new, base1, base2)
        print(f"  [⑤] After spatial : {'✅ clear' if free else '❌ still colliding'}"
              f"  min = {min_d * 100:.1f} cm")

        if free:
            report = _make_report(True, iteration, min_d,
                                   f"Resolved by spatial refinement at iteration {iteration}")
            print(f"\n✅  Collision-free after {iteration} iteration(s)  "
                  f"(spatial refinement resolved it)  min_clr={min_d * 100:.1f} cm")
            return t1_new, t2_new, report

        traj1, traj2 = t1_new, t2_new

    
    print(f"\n{'=' * 80}")
    print(f"❌ FAILED — collision not resolved in {Config.MAX_ITER} iterations")
    print(f"   Best min clearance : {min_d * 100:.1f} cm  "
          f"(required {Config.MIN_SAFE_DIST * 100:.0f} cm)")
    print("=" * 80)
    return None, None, _make_report(
        False, Config.MAX_ITER, min_d,
        f"FAILED after {Config.MAX_ITER} iterations — choose different targets")


def _make_report(success: bool, iterations: int,
                  min_clearance: float, message: str) -> Dict:
    return {
        'success':          success,
        'iterations_used':  iterations,
        'min_clearance_m':  float(min_clearance),
        'min_clearance_cm': round(float(min_clearance) * 100, 2),
        'required_cm':      Config.MIN_SAFE_DIST * 100,
        'message':          message,
    }




def _traj_to_output(traj: Dict) -> Dict:
    
    t   = traj['time']
    pos = traj['positions']
    vel = traj['velocities']
    acc = traj['accelerations']

    def _to_list(v):
        return v.tolist() if isinstance(v, np.ndarray) else v

   
    traj_pts = [
        {'time': float(t[i]), 'joints': pos[i].tolist()}
        for i in range(len(t))
    ]

    return {
        
        'trajectory_points': traj_pts,

        
        'time':            _to_list(t),
        'positions':       _to_list(pos),
        'velocities':      _to_list(vel),
        'accelerations':   _to_list(acc),
        'num_samples':     traj['num_samples'],

        
        'joint_waypoints': _to_list(traj.get('joint_waypoints',
                                              pos[[0, -1]])),
        'waypoint_times':  _to_list(traj.get('waypoint_times',
                                              np.array([0.0, t[-1]]))),
    }


def load_trajectories() -> Tuple[Optional[Dict], Optional[Dict]]:
    
    if not os.path.exists('trajectories.json'):
        print("❌  trajectories.json not found")
        return None, None

    with open('trajectories.json') as f:
        data = json.load(f)

    def _parse(arm_data: Dict, arm_id: str) -> Dict:
        
        if 'time' in arm_data:
            src = arm_data
        
        elif 'trajectory' in arm_data:
            src = arm_data['trajectory']
        else:
            raise ValueError(f"{arm_id}: cannot find time/positions arrays")

        pos_arr = np.array(src['positions'],     dtype=float)
        t_arr   = np.array(src['time'],          dtype=float)
        vel_arr = np.array(src.get('velocities',
                     np.gradient(pos_arr, t_arr, axis=0)), dtype=float)
        acc_arr = np.array(src.get('accelerations',
                     np.gradient(vel_arr, t_arr, axis=0)), dtype=float)

        
        raw_wps = arm_data.get('joint_waypoints',
                  src.get('joint_waypoints', None))
        raw_t   = arm_data.get('waypoint_times',
                  src.get('waypoint_times',  None))

        if raw_wps is not None and len(raw_wps) >= 2:
            wps = np.array(raw_wps, dtype=float)
            t_w = np.array(raw_t,   dtype=float) if raw_t is not None \
                  else np.linspace(0.0, float(t_arr[-1]), len(wps))
        else:
            
            meta = arm_data.get('metadata', {})
            s = np.array(meta.get('start_joints', pos_arr[0].tolist()), dtype=float)
            e = np.array(meta.get('end_joints',   pos_arr[-1].tolist()), dtype=float)
            wps = np.vstack([s, e])
            t_w = np.array([0.0, float(t_arr[-1])])

        return {
            'time':            t_arr,
            'positions':       pos_arr,
            'velocities':      vel_arr,
            'accelerations':   acc_arr,
            'num_samples':     len(t_arr),
            'joint_waypoints': wps,
            'waypoint_times':  t_w,
            'metadata':        arm_data.get('metadata', {}),
        }

    t1 = _parse(data.get('dsr01', {}), 'dsr01')
    t2 = _parse(data.get('dsr02', {}), 'dsr02')

    print(f"✓  trajectories.json loaded")
    print(f"   dsr01 : {len(t1['positions'])} samples  "
          f"T={t1['time'][-1]:.3f}s  waypoints={len(t1['joint_waypoints'])}")
    print(f"   dsr02 : {len(t2['positions'])} samples  "
          f"T={t2['time'][-1]:.3f}s  waypoints={len(t2['joint_waypoints'])}")
    return t1, t2



def main():
    print("\n" + "=" * 80)
    print("KURAMOTO SYNCHRONIZATION  (Paper §IV — External Sync)")
    print("=" * 80)
    print("  Reads  : collision_result.json  +  trajectories.json")
    print("  Writes : synchronized_trajectories.json  |  fail_report.json")
    print("=" * 80)

    
    col_file = 'collision_result.json'
    if not os.path.exists(col_file):
        print(f"\n❌  {col_file} not found")
        print("    Run: ros2 run dual_arm_sync collision_checker")
        return

    with open(col_file) as f:
        cr = json.load(f)

    
    collision_detected = not cr.get('collision_free', True)
    if collision_detected:
        print(f"\n⚠   Collision detected by checker  "
              f"(max_pen={cr.get('max_penetration_m', 0)*1000:.2f} mm  "
              f"conflicting={cr.get('conflicting_arms', [])})")
        print("    Kuramoto will attempt resolution...")
    else:
        print(f"\n✓   Checker reports collision-free  "
              f"(min_clr={cr.get('min_clr_pair_m', {})})  "
              f"— running Kuramoto for phase alignment anyway")

    
    traj1, traj2 = load_trajectories()
    if traj1 is None or traj2 is None:
        return

    base1 = RobotBases.DSR01_BASE
    base2 = RobotBases.DSR02_BASE

    
    time_offsets = cr.get('time_offsets', {'dsr01': 0.0, 'dsr02': 0.0})
    if any(abs(v) > 1e-6 for v in time_offsets.values()):
        print(f"\n  Collision-checker phase seeds: {time_offsets}")
        T = float(traj1['time'][-1])
        for arm_id, offset in time_offsets.items():
            if abs(offset) < 1e-6:
                continue
            traj = traj1 if arm_id == 'dsr01' else traj2
            wps   = traj['joint_waypoints']
            t_wps = traj['waypoint_times']
            
            t_new = np.concatenate([[0.0], t_wps + offset])
            t_new = np.clip(t_new, 0.0, T + offset)
            w_new = np.vstack([wps[0:1], wps])
            updated = rebuild_bspline(w_new, t_new / (T + offset), T + offset)
            if arm_id == 'dsr01':
                traj1 = updated
            else:
                traj2 = updated
            print(f"  Applied seed offset to {arm_id}: {offset:.4f}s")

   
    t_start = time.time()
    t1_sync, t2_sync, report = synchronize(traj1, traj2, base1, base2)
    elapsed = time.time() - t_start
    report['elapsed_s'] = round(elapsed, 2)

    
    print("\n" + "=" * 80)

    if report['success']:
        output = {
            'arm_ids':  ['dsr01', 'dsr02'],
            'collision_free': True,        
            'dsr01':    _traj_to_output(t1_sync),
            'dsr02':    _traj_to_output(t2_sync),
            'synchronization_report': report,
            'parameters': {
                'min_safe_distance_m':  Config.MIN_SAFE_DIST,
                'base_coupling':        Config.BASE_COUPLING,
                'max_coupling':         Config.MAX_COUPLING,
                'repulsion_strength':   Config.REPULSION_STR,
                'emergency_strength':   Config.EMERGENCY_STR,
                'max_iterations':       Config.MAX_ITER,
                'perturb_scale_0_rad':  Config.PERTURB_SCALE_0,
                'anneal_rate':          Config.ANNEAL_RATE,
            },
        }
        with open('synchronized_trajectories.json', 'w') as f:
            json.dump(output, f, indent=2)
        kb = os.path.getsize('synchronized_trajectories.json') / 1024
        print("RESULT : ✅  SUCCESS")
        print(f"  Iterations    : {report['iterations_used']}")
        print(f"  Min clearance : {report['min_clearance_cm']:.1f} cm")
        print(f"  Elapsed       : {elapsed:.1f}s")
        print(f"\n✓  Saved: synchronized_trajectories.json  ({kb:.0f} KB)")
        print("\nNext step: ros2 run dual_arm_sync gazebo_executor")

    else:
        fail_out = {
            'status': 'FAILED',
            'collision_free': False,
            'report':  report,
            'message': report['message'],
        }
        with open('fail_report.json', 'w') as f:
            json.dump(fail_out, f, indent=2)
        print("RESULT : ❌  FAILED")
        print(f"  Iterations    : {report['iterations_used']} / {Config.MAX_ITER}")
        print(f"  Best clearance: {report['min_clearance_cm']:.1f} cm "
              f"(required {Config.MIN_SAFE_DIST * 100:.0f} cm)")
        print(f"  Elapsed       : {elapsed:.1f}s")
        print("\n✗  Saved: fail_report.json")
        print("\n⚠   Action: choose different target positions and re-run pipeline")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()