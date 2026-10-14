#!/usr/bin/env python3

import time
import numpy as np

try:
    from dual_arm_sync.dual_arm_sync import multi_arm_core as C
except Exception:
    import multi_arm_core as C

NAMES = C.ROBOT_NAMES
BASES = C.ROBOT_BASES
NDOF = C.NDOF


def _ee_path_len(pos, base):
    ee = np.array([C.fk_world(pos[k], base) for k in range(len(pos))])
    return float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1))) if len(ee) > 1 else 0.0


def plan_trial(arms_data, start_qs, duration):
    
    t_total = time.time()
    timings = {}
    norm_start = {n: C.normalize_joints(start_qs.get(n, np.zeros(NDOF))) for n in NAMES}

    result = {
        'outcome': None, 'cp_rounds': 0, 'had_collision': False,
        'n_coll_raw': 0, 'n_coll_final': 0, 'arm_pos': None, 'duration': duration,
        'target_joints': {}, 'start_joints': {n: norm_start[n].tolist() for n in NAMES},
        'inter_arm_clear_cm': None, 'kur_min_dist_cm': None,
        'ee_path_length_m': {}, 'timings': timings, 'plan_time_s': None,
    }

    # ---- IK ----
    t0 = time.time()
    arms = {n: {'base': arms_data[n]['base'], 'tloc': arms_data[n]['tloc'],
                'trot': arms_data[n]['trot']} for n in NAMES}
    best = C.find_best_targets(arms, norm_start)
    timings['ik_s'] = round(time.time() - t0, 4)
    if best is None:
        result['outcome'] = 'FAIL_IK'
        result['plan_time_s'] = round(time.time() - t_total, 3)
        return result
    for n in NAMES:
        result['target_joints'][n] = best[n].tolist()
    clears = [C.pair_min_dist(best[NAMES[i]], BASES[NAMES[i]], best[NAMES[j]], BASES[NAMES[j]])
              for i in range(len(NAMES)) for j in range(i + 1, len(NAMES))]
    result['inter_arm_clear_cm'] = round(min(clears) * 100, 2)

    
    t0 = time.time()
    res = C.plan_arms(norm_start, best, {n: arms_data[n]['base'] for n in NAMES}, verbose=False)
    timings['plan_s'] = round(time.time() - t0, 4)

    pos = res['positions']
    for n in NAMES:
        result['ee_path_length_m'][n] = round(_ee_path_len(pos[n], BASES[n]), 4)

   
    seed_pos = {n: C.sample_minjerk(C.seed_spline(norm_start[n], best[n]),
                                    max(2, int(res['duration'] * C.RATE_HZ))) for n in NAMES}
    result['n_coll_raw'] = _count_coll(seed_pos)
    result['had_collision'] = result['n_coll_raw'] > 0
    result['n_coll_final'] = res['residual_collision_steps']
    result['duration'] = res['duration']
    result['arm_pos'] = {n: np.asarray(pos[n], float) for n in NAMES}

    
    Kp = min(len(pos[n]) for n in NAMES)
    kmin = min(C.pair_min_dist(pos[NAMES[i]][k], BASES[NAMES[i]],
                               pos[NAMES[j]][k], BASES[NAMES[j]])
               for k in range(Kp)
               for i in range(len(NAMES)) for j in range(i + 1, len(NAMES)))
    result['kur_min_dist_cm'] = round(kmin * 100, 2)

    
    oc = res['outcome']
    if oc.startswith('RESOLVED_CP_'):
        oc = 'RESOLVED_CP_%d' % min(int(oc.rsplit('_', 1)[1]), 5)
    result['outcome'] = oc
    result['cp_rounds'] = min(int(res['rounds']), 5)
    result['plan_time_s'] = round(time.time() - t_total, 3)
    return result


def _count_coll(arm_pos):
    K = min(len(arm_pos[n]) for n in NAMES)
    nc = 0
    for k in range(K):
        hit = False
        for i in range(len(NAMES)):
            for j in range(i + 1, len(NAMES)):
                if C.pair_collides(arm_pos[NAMES[i]][k], BASES[NAMES[i]],
                                   arm_pos[NAMES[j]][k], BASES[NAMES[j]]):
                    hit = True; break
            if hit:
                break
        if hit:
            nc += 1
    return nc