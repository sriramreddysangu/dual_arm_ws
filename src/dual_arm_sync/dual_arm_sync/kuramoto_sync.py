#!/usr/bin/env python3

import json
import numpy as np
import warnings
from typing import Dict, List, Optional, Set, Tuple

warnings.filterwarnings('ignore')

try:
    from dual_arm_sync.ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
    )
    from dual_arm_sync.trajectory_generation import refit_arm_trajectory
    from dual_arm_sync.collision_checker import check_trajectories, _resolve_bases
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
    )
    from trajectory_generation import refit_arm_trajectory
    from collision_checker import check_trajectories, _resolve_bases


K_GLOBAL        = 5.0   
K_LOCAL         = 8.0   
SYNC_DT         = 0.01  
MAX_SYNC_TIME   = 30.0  
CONV_THRESHOLD  = 0.05  
OUTPUT_STEPS    = 200    

MAX_REFINE_ITER = 3     
REFINE_TRIGGER  = 0.005 
CP_INCREMENT    = 2     
CP_PER_SEG      = 4   
N_CP_MAX_REFINE = 24    
N_SEG_MAX_REFINE = 12    


def build_arrays(trajectory_points: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    
    t_arr = np.array([p['time']   for p in trajectory_points])
    j_arr = np.array([p['joints'] for p in trajectory_points])
    return t_arr, j_arr


def interpolate_at(t_arr: np.ndarray,
                   j_arr: np.ndarray,
                   t_q:   float) -> np.ndarray:
    
    t_q = float(np.clip(t_q, t_arr[0], t_arr[-1]))
    idx = np.searchsorted(t_arr, t_q)
    if idx == 0:
        return j_arr[0].copy()
    if idx >= len(t_arr):
        return j_arr[-1].copy()
    alpha = (t_q - t_arr[idx-1]) / (t_arr[idx] - t_arr[idx-1] + 1e-12)
    return j_arr[idx-1] + alpha * (j_arr[idx] - j_arr[idx-1])


def phase_spread(phases: np.ndarray) -> float:
    
    if len(phases) <= 1:
        return 0.0
    R = abs(np.mean(np.exp(1j * phases)))
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))


def _kuramoto_step(phases: np.ndarray,
                   omegas: np.ndarray,
                   K:      float,
                   dt:     float) -> np.ndarray:
    
    N    = len(phases)
    dphi = omegas.copy()
    for i in range(N):
        dphi[i] += (K / N) * sum(
            np.sin(phases[j] - phases[i]) for j in range(N) if j != i
        )
    return phases + dphi * dt


def run_kuramoto(arm_ids:    List[str],
                 durations:  Dict[str, float],
                 offsets:    Dict[str, float],
                 K:          float  = K_GLOBAL,
                 dt:         float  = SYNC_DT,
                 max_time:   float  = MAX_SYNC_TIME,
                 label:      str    = 'global'
                 ) -> Tuple[Dict[str, np.ndarray], bool, float]:
    
    N      = len(arm_ids)
    omegas = np.array([2.0 * np.pi / durations[a] for a in arm_ids])
    phases = np.array([2.0 * np.pi * offsets.get(a, 0.0) / durations[a]
                       for a in arm_ids])

    n_max    = int(max_time / dt)
    history  = {a: [] for a in arm_ids}
    log_every = max(1, n_max // 8)

    print(f'\n  [Kuramoto {label}]  arms={arm_ids}  N={N}  K={K}  dt={dt}')
    print(f'  ω  : {dict(zip(arm_ids, np.round(omegas, 4)))}')
    print(f'  φ₀ : {dict(zip(arm_ids, np.round(phases, 4)))}')
    print(f'  Δt : {offsets}')

    converged = False
    for step in range(n_max):
        for i, a in enumerate(arm_ids):
            history[a].append(float(phases[i]))

        spread = phase_spread(phases)
        if step % log_every == 0:
            print(f'    step={step:5d}  t={step*dt:.2f}s'
                  f'  spread={spread:.4f}rad'
                  f'  φ={np.round(phases % (2*np.pi), 3)}')

        if spread < CONV_THRESHOLD and step > 5:
            converged = True
            print(f'\n  ✓  [{label}] Converged  step={step}'
                  f'  t={step*dt:.2f}s  spread={spread:.5f}rad')
            break

        phases = _kuramoto_step(phases, omegas, K, dt)

    if not converged:
        print(f'\n  ⚠  [{label}] Not converged within {max_time}s'
              f'  spread={phase_spread(phases):.4f}rad')

    return ({a: np.array(history[a]) for a in arm_ids},
            converged, phase_spread(phases))


def phase_to_traj(arm_id:        str,
                   phase_hist:    np.ndarray,
                   t_arr:         np.ndarray,
                   j_arr:         np.ndarray,
                   duration:      float,
                   dt:            float = SYNC_DT,
                   out_steps:     int   = OUTPUT_STEPS) -> Dict:
    

    stable      = min(30, max(1, len(phase_hist)))
    phi_stable  = float(np.mean(phase_hist[-stable:]))
    phi_conv    = phi_stable % (2.0 * np.pi)
    t_off       = phi_conv / (2.0 * np.pi) * duration   

    T_traj  = float(t_arr[-1] - t_arr[0])              
    t_wall  = np.linspace(0.0, duration, out_steps)
    t_eff_arr = np.empty(out_steps)

    for i, tw in enumerate(t_wall):
        if t_off > 1e-6 and tw < t_off:
           
            t_eff_arr[i] = t_arr[0]
        else:
            
            play_elapsed  = tw - t_off
            play_window   = duration - t_off
            if play_window > 1e-6:
                frac = play_elapsed / play_window
            else:
                frac = 1.0
            t_eff_arr[i] = t_arr[0] + frac * T_traj

    t_eff_arr = np.clip(t_eff_arr, t_arr[0], t_arr[-1])

    points = [{'time':     float(tw),
               'time_eff': float(te),
               'joints':   interpolate_at(t_arr, j_arr, te).tolist()}
              for tw, te in zip(t_wall, t_eff_arr)]

    return {
        'phase_history':     phase_hist.tolist(),
        'time_history':      t_eff_arr.tolist(),
        'trajectory_points': points,
        't_offset_applied':  float(t_off),
    }

def _check_in_memory(trajs:     Dict,
                      arm_bases: Dict[str, np.ndarray]) -> Dict:
    
    return check_trajectories(trajs, arm_bases)


def _increment_params(n_seg_cur: int, n_cp_cur: int, iteration: int
                      ) -> Tuple[int, int]:
    
    n_cp  = min(N_CP_MAX_REFINE, n_cp_cur + CP_INCREMENT)
    n_seg = min(N_SEG_MAX_REFINE,
                max(n_seg_cur, int(np.ceil(n_cp / CP_PER_SEG))))
    return n_seg, n_cp


def adaptive_refinement_loop(trajs_init:    Dict[str, Dict],
                               ik_data:       Dict,
                               arm_bases:     Dict[str, np.ndarray],
                               durations:     Dict[str, float],
                               offsets_init:  Dict[str, float]
                               ) -> Tuple[Dict, Dict, List[Dict]]:
    
    arm_ids    = list(trajs_init.keys())
    trajs      = {a: dict(t) for a, t in trajs_init.items()}
    offsets    = dict(offsets_init)
    refine_log: List[Dict] = []

    
    n_seg_cur  = {a: trajs[a]['n_seg'] for a in arm_ids}
    n_cp_cur   = {a: trajs[a]['n_cp']  for a in arm_ids}

    print(f'\n{"="*80}')
    print('ADAPTIVE REFINEMENT LOOP')
    print(f'{"="*80}')
    print(f'  Max iterations    : {MAX_REFINE_ITER}')
    print(f'  Refine trigger    : {REFINE_TRIGGER*1000:.1f} mm penetration')
    print(f'  CP increment      : +{CP_INCREMENT} per conflicting arm')
    print(f'  Initial N_seg/N_cp: '
          f'{dict((a, (n_seg_cur[a], n_cp_cur[a])) for a in arm_ids)}')

    
    print(f'\n{"─"*70}')
    print('ITERATION 0  —  Global Kuramoto')
    print(f'{"─"*70}')

    phase_hists, converged, spread = run_kuramoto(
        arm_ids, durations, offsets,
        K=K_GLOBAL, dt=SYNC_DT, max_time=MAX_SYNC_TIME,
        label='global-0',
    )

    
    sync_trajs: Dict[str, Dict] = {}
    arr_cache:  Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for a in arm_ids:
        t_arr, j_arr = build_arrays(trajs[a]['trajectory_points'])
        arr_cache[a] = (t_arr, j_arr)
        sync_trajs[a] = phase_to_traj(
            a, phase_hists[a], t_arr, j_arr, durations[a]
        )

    
    coll = _check_in_memory(sync_trajs, arm_bases)
    max_pen = coll['max_penetration_m']

    log0 = {
        'iteration':              0,
        'phase':                  'global_kuramoto',
        'max_pen_mm':             round(max_pen * 1000, 3),
        'safe':                   coll['safe'],
        'collision_free':         coll['safe'],
        'collision_free_summary': coll.get('collision_free_summary',
                                   'COLLISION-FREE' if coll['safe'] else
                                   f'COLLISION max_pen={max_pen*1000:.2f}mm'),
        'converged':              converged,
        'spread_rad':             round(spread, 5),
        'conflicting':            coll['conflicting_arms'],
        'n_seg':                  dict(n_seg_cur),
        'n_cp':                   dict(n_cp_cur),
    }
    refine_log.append(log0)
    print(f'\n  [0] max_pen={max_pen*1000:.2f}mm  safe={coll["safe"]}'
          f'  converged={converged}  spread={spread:.4f}rad')

    
    for it in range(1, MAX_REFINE_ITER + 1):
        
        if coll['safe'] or max_pen <= REFINE_TRIGGER:
            print(f'\n  ✓  Collision resolved before iteration {it}'
                  f'   max_pen={max_pen*1000:.2f}mm ≤ {REFINE_TRIGGER*1000:.1f}mm')
            break

        conflicting: List[str] = coll['conflicting_arms']
        if not conflicting:
            print(f'\n  ⚠  No conflicting arms identified — stopping')
            break

        print(f'\n{"─"*70}')
        print(f'ITERATION {it}  —  Spatial refinement + local Kuramoto')
        print(f'{"─"*70}')
        print(f'  Conflicting arms : {conflicting}')
        print(f'  max penetration  : {max_pen*1000:.2f} mm')

       
        for a in conflicting:
            old_seg, old_cp = n_seg_cur[a], n_cp_cur[a]
            n_seg_cur[a], n_cp_cur[a] = _increment_params(
                n_seg_cur[a], n_cp_cur[a], it
            )
            print(f'  [{a}]  N_seg: {old_seg}→{n_seg_cur[a]}'
                  f'   N_cp: {old_cp}→{n_cp_cur[a]}')

        
        for a in conflicting:
            arm_ik = ik_data.get(a, {})
            
            arm_ik['arm_id'] = a
            refitted = refit_arm_trajectory(
                arm_ik, n_seg_cur[a], n_cp_cur[a], durations[a]
            )
            if refitted:
                trajs[a]          = refitted
                t_arr, j_arr      = build_arrays(refitted['trajectory_points'])
                arr_cache[a]      = (t_arr, j_arr)
                print(f'  [{a}]  seed refit  OK  '
                      f'steps={len(refitted["trajectory_points"])}')
            else:
                print(f'  [{a}]  ⚠  seed refit FAILED — keeping previous')

        
        local_offsets = {a: 0.0 for a in conflicting}
        for a in conflicting:
            
            local_offsets[a] = offsets.get(a, 0.0)

        local_hists, local_conv, local_spread = run_kuramoto(
            conflicting,
            durations,
            local_offsets,
            K        = K_LOCAL,
            dt       = SYNC_DT,
            max_time = MAX_SYNC_TIME / 2,
            label    = f'local-{it}',
        )

        
        for a in arm_ids:
            t_arr, j_arr = arr_cache[a]
            if a in conflicting:
                sync_trajs[a] = phase_to_traj(
                    a, local_hists[a], t_arr, j_arr, durations[a]
                )
            else:
                sync_trajs[a] = phase_to_traj(
                    a, phase_hists[a], t_arr, j_arr, durations[a]
                )

        
        coll    = _check_in_memory(sync_trajs, arm_bases)
        max_pen = coll['max_penetration_m']

        log_i = {
            'iteration':              it,
            'phase':                  'spatial_refine + local_kuramoto',
            'max_pen_mm':             round(max_pen * 1000, 3),
            'safe':                   coll['safe'],
            'collision_free':         coll['safe'],
            'collision_free_summary': coll.get('collision_free_summary',
                                       'COLLISION-FREE' if coll['safe'] else
                                       f'COLLISION max_pen={max_pen*1000:.2f}mm'),
            'local_converged':        local_conv,
            'local_spread':           round(local_spread, 5),
            'conflicting':            coll['conflicting_arms'],
            'n_seg':                  dict(n_seg_cur),
            'n_cp':                   dict(n_cp_cur),
        }
        refine_log.append(log_i)
        print(f'\n  [{it}] max_pen={max_pen*1000:.2f}mm  safe={coll["safe"]}'
              f'  local_converged={local_conv}'
              f'  local_spread={local_spread:.4f}rad')

        
        offsets = coll['time_offsets']

        
        for a in conflicting:
            phase_hists[a] = local_hists[a]

    final_collision_free = coll['safe'] or max_pen <= COLLISION_TOL

    return trajs, sync_trajs, refine_log, converged, spread, final_collision_free



def run(traj_file:      str = 'trajectories.json',
        ik_file:        str = 'ik_solutions.json',
        collision_file: str = 'collision_result.json',
        output_file:    str = 'synchronized_trajectories.json') -> Dict:
    
    print('\n' + '=' * 80)
    print('KURAMOTO SYNC  [Global + Adaptive Spatial Refinement]')
    print('=' * 80)

    
    with open(traj_file) as f:
        traj_data = json.load(f)

    trajs    = traj_data.get('trajectories', traj_data)
    arm_ids  = traj_data.get('arm_ids', list(trajs.keys()))
    duration = float(traj_data.get('duration', 10.0))
    durations = {a: float(trajs[a].get('duration', duration)) for a in arm_ids}

    print(f'\n  Arms     : {arm_ids}')
    print(f'  Duration : {duration} s')
    print(f'  N_seg    : {dict((a, trajs[a]["n_seg"]) for a in arm_ids)}')
    print(f'  N_cp     : {dict((a, trajs[a]["n_cp"])  for a in arm_ids)}')

    
    ik_data: Dict = {}
    try:
        with open(ik_file) as f:
            ik_raw = json.load(f)
        for a in arm_ids:
            if a in ik_raw:
                ik_data[a] = ik_raw[a]
        print(f'\n  IK data loaded : {list(ik_data.keys())}')
    except FileNotFoundError:
        print(f'\n  ⚠  {ik_file} not found — refinement refit disabled')

    
    time_offsets: Dict[str, float] = {a: 0.0 for a in arm_ids}
    initial_conflicting: List[str] = []
    try:
        with open(collision_file) as f:
            coll_init = json.load(f)
        time_offsets.update(coll_init.get('time_offsets', {}))
        initial_conflicting = coll_init.get('conflicting_arms', [])
        safe_init           = coll_init.get('safe', True)
        print(f'\n  Initial collision : safe={safe_init}'
              f'  conflicting={initial_conflicting}')
        if any(v > 0 for v in time_offsets.values()):
            print(f'  Phase offsets    : {time_offsets}')
        else:
            print('  Phase offsets    : none (all arms start in phase)')
    except FileNotFoundError:
        print(f'\n  ⚠  {collision_file} not found — no initial offsets')

    
    arm_bases = _resolve_bases(arm_ids)

    
    final_trajs, sync_trajs, refine_log, global_conv, global_spread, \
        final_cf = adaptive_refinement_loop(
            trajs_init   = trajs,
            ik_data      = ik_data,
            arm_bases    = arm_bases,
            durations    = durations,
            offsets_init = time_offsets,
        )

    
    print(f'\n{"="*80}')
    print('REFINEMENT SUMMARY')
    print(f'{"="*80}')
    for log in refine_log:
        print(f'  iter={log["iteration"]}  '
              f'max_pen={log["max_pen_mm"]:.2f}mm  '
              f'safe={log["safe"]}  '
              f'phase={log["phase"]}')
        print(f'    N_seg={log["n_seg"]}  N_cp={log["n_cp"]}')

    print(f'\n  global_converged  : {global_conv}')
    print(f'  global_spread     : {global_spread:.5f} rad')
    print(f'  final_collision_free : {final_cf}')
    print(f'  refinement_iters  : {len(refine_log)-1}')

    
    traj_output: Dict[str, Dict] = {}
    for a in arm_ids:
        sync = sync_trajs[a]
        traj_output[a] = {
            'phase_history':     sync['phase_history'],
            'time_history':      sync['time_history'],
            'trajectory_points': sync['trajectory_points'],
            'n_seg_final':       final_trajs[a]['n_seg'],
            'n_cp_final':        final_trajs[a]['n_cp'],
            'method':            final_trajs[a].get('method', 'kabir_2019'),
        }
        print(f'  {a}: steps={len(sync["trajectory_points"])}'
              f'  N_seg={final_trajs[a]["n_seg"]}'
              f'  N_cp={final_trajs[a]["n_cp"]}')

    output = {
        'arm_ids':               arm_ids,
        'converged':             global_conv,
        'final_spread_rad':      float(global_spread),
        'collision_free':        final_cf,
        'refinement_iterations': len(refine_log) - 1,
        'refinement_log':        refine_log,
        'k_global':              K_GLOBAL,
        'k_local':               K_LOCAL,
        'dt':                    SYNC_DT,
        'original_duration':     duration,
        'time_offsets_used':     time_offsets,
        'trajectories':          traj_output,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n✓  {output_file}  written')
    print('Next → ros2 run dual_arm_sync gazebo_executor')
    return output


def main(args=None):
    try:
        import rclpy; rclpy.init(args=args)
    except Exception:
        pass
    try:
        run()
    except FileNotFoundError as e:
        print(f'✗  {e}')
        print('   Run trajectory_generation and collision_checker first')
    except Exception:
        import traceback; traceback.print_exc()
    try:
        import rclpy; rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()