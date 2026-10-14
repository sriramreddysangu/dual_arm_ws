#!/usr/bin/env python3


from __future__ import annotations

import json
import os
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


try:
    from dual_arm_sync.ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
        get_sphere_centres,
        L1, L2, L3, L4, A,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
        get_sphere_centres,
        L1, L2, L3, L4, A,
    )




SAFETY_MARGIN = 0.15  
WARN_MARGIN   = 0.08  
EVAL_EVERY_N  = 1      



def _z(v):
    """Local z-axis point."""
    return np.array([0., 0., float(v)])

def _x(v):
    """Local x-axis point."""
    return np.array([float(v), 0., 0.])

def _o():
    """Origin."""
    return np.array([0., 0., 0.])


M1013_CAPSULES = [
    
    (0, _o(),       _z(L1),          0.095),

    
    (1, _o(),       _x(L2 * 0.45),   0.080), 
    (1, _x(L2*0.45), _x(L2),         0.070),  

    
    (2, _o(),       _x(A + 0.04),    0.070),  

    
    (3, _o(),       _z(L3 * 0.50),   0.065),   
    (3, _z(L3*0.50), _z(L3),         0.055),  

    
    (4, _o(),       _o(),            0.055),

    
    (5, _o(),       _z(L4),          0.045),
]




def _try_load_urdf_capsules():
    
    try:
        import subprocess
        import xml.etree.ElementTree as ET

        
        res = subprocess.run(
            ['ros2', 'pkg', 'prefix', 'dsr_description2'],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode != 0:
            return None

        prefix = res.stdout.strip()

        
        candidates = [
            os.path.join(prefix, 'share', 'dsr_description2',
                         'xacro', 'm1013.urdf.xacro'),
            os.path.normpath(os.path.join(
                prefix, '..', 'src', 'dsr_description2',
                'xacro', 'm1013.urdf.xacro')),
        ]
        xp = next((p for p in candidates if os.path.isfile(p)), None)
        if xp is None:
            return None

        
        xr = subprocess.run(
            ['xacro', xp, 'color:=white', 'gripper:=none'],
            capture_output=True, text=True, timeout=15,
        )
        if xr.returncode != 0:
            return None

        root = ET.fromstring(xr.stdout)

        
        link_to_frame = {
            'base_link': 0, 'link_1': 0,
            'link_2': 1, 'link_3': 2,
            'link_4': 3, 'link_5': 4, 'link_6': 5,
        }

        capsules = []
        for link in root.findall('link'):
           
            bare = link.get('name', '').split('/')[-1]
            fidx = link_to_frame.get(bare)
            if fidx is None:
                continue

            for coll in link.findall('collision'):
                geom = coll.find('geometry')
                if geom is None:
                    continue

                
                orig = coll.find('origin')
                ox = oy = oz = 0.0
                if orig is not None:
                    xyz = orig.get('xyz', '0 0 0').split()
                    ox, oy, oz = float(xyz[0]), float(xyz[1]), float(xyz[2])

                cyl = geom.find('cylinder')
                box = geom.find('box')
                sph = geom.find('sphere')

                if cyl is not None:
                    r  = float(cyl.get('radius', '0.05'))
                    hl = float(cyl.get('length',  '0.1')) / 2.0
                    
                    capsules.append((fidx,
                                     np.array([ox, oy, oz - hl]),
                                     np.array([ox, oy, oz + hl]),
                                     r + 0.005))

                elif box is not None:
                    sz  = [float(v) for v in
                           box.get('size', '0.1 0.1 0.1').split()]
                    ax  = int(np.argmax(sz))  
                    hl  = sz[ax] / 2.0
                    r   = max(sz[i] for i in range(3) if i != ax) / 2.0
                    p0  = np.array([ox, oy, oz])
                    p1  = np.array([ox, oy, oz])
                    p0[ax] -= hl
                    p1[ax] += hl
                    capsules.append((fidx, p0, p1, r + 0.005))

                elif sph is not None:
                    r  = float(sph.get('radius', '0.05'))
                    p  = np.array([ox, oy, oz])
                    capsules.append((fidx, p, p.copy(), r + 0.005))

        if len(capsules) >= 4:
            print(f'  [URDF loader]  {len(capsules)} capsules parsed from {xp}')
            return capsules

    except Exception as exc:
        warnings.warn(f'URDF capsule loader skipped ({type(exc).__name__}: {exc})')
    return None


def _load_capsules():
    urdf = _try_load_urdf_capsules()
    return urdf if urdf else M1013_CAPSULES


CAPSULES = _load_capsules()
N_CAP    = len(CAPSULES)




def _seg_to_seg_dist(p0, p1, q0, q1):
    
    d1 = p1 - p0;  d2 = q1 - q0;  r = p0 - q0
    a  = float(np.dot(d1, d1))
    e  = float(np.dot(d2, d2))
    f  = float(np.dot(d2, r))
    EPS = 1e-10

    if a < EPS and e < EPS:          
        return float(np.linalg.norm(r))
    if a < EPS:                      
        s, t = 0.0, float(np.clip(f / (e + EPS), 0, 1))
    else:
        c = float(np.dot(d1, r))
        if e < EPS:                 
            t, s = 0.0, float(np.clip(-c / (a + EPS), 0, 1))
        else:
            b   = float(np.dot(d1, d2))
            den = a * e - b * b
            s   = float(np.clip((b * f - c * e) / den, 0, 1)) \
                  if abs(den) > EPS else 0.0
            t   = (b * s + f) / (e + EPS)
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / (a + EPS), 0, 1))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / (a + EPS), 0, 1))

    return float(np.linalg.norm(p0 + s * d1 - (q0 + t * d2)))


def _caps_world(joints, base):
    
    _, _, frames = forward_kinematics(joints)
    out = []
    for (fi, p0l, p1l, r) in CAPSULES:
        T  = frames[fi]
        R  = T[:3, :3]
        t  = T[:3,  3]
        out.append((R @ p0l + t + base,
                    R @ p1l + t + base,
                    r))
    return out


def _caps_penetration(caps_a, caps_b):
    
    total = 0.0
    for (p0a, p1a, ra) in caps_a:
        for (p0b, p1b, rb) in caps_b:
            total += max(0.0, ra + rb - _seg_to_seg_dist(p0a, p1a, p0b, p1b))
    return total


def _caps_min_clearance(caps_a, caps_b):
    
    mn = np.inf
    for (p0a, p1a, ra) in caps_a:
        for (p0b, p1b, rb) in caps_b:
            clr = _seg_to_seg_dist(p0a, p1a, p0b, p1b) - ra - rb
            if clr < mn:
                mn = clr
    return float(mn)


def _self_pen_caps(joints):
    
    _, _, frames = forward_kinematics(joints)
    local_caps = []
    for (fi, p0l, p1l, r) in CAPSULES:
        T  = frames[fi]
        R  = T[:3, :3];  t = T[:3, 3]
        local_caps.append((R @ p0l + t, R @ p1l + t, r))

    total = 0.0
    for i in range(N_CAP):
        for j in range(i + 2, N_CAP):
            if abs(CAPSULES[i][0] - CAPSULES[j][0]) <= 2:
                continue   
            d = _seg_to_seg_dist(local_caps[i][0], local_caps[i][1],
                                  local_caps[j][0], local_caps[j][1])
            total += max(0.0, CAPSULES[i][3] + CAPSULES[j][3] - d)
    return total




def _centres_spheres(joints, base):
    _, _, frames = forward_kinematics(joints)
    c = np.zeros((N_SPHERES, 3))
    for k, (fi, zo, _) in enumerate(LINK_SPHERES):
        c[k] = frames[fi][:3, 3] + zo * frames[fi][:3, 2] + base
    return c

def _sphere_pen(c1, c2):
    total = 0.0
    for i in range(N_SPHERES):
        for j in range(N_SPHERES):
            d = np.linalg.norm(c1[i] - c2[j])
            total += max(0.0, LINK_SPHERES[i][2] + LINK_SPHERES[j][2] - d)
    return total

def _sphere_min_clr(c1, c2):
    mn = np.inf
    for i in range(N_SPHERES):
        for j in range(N_SPHERES):
            d = np.linalg.norm(c1[i]-c2[j]) - LINK_SPHERES[i][2] - LINK_SPHERES[j][2]
            if d < mn: mn = d
    return float(mn)



def check_trajectories(trajectories, arm_bases, use_capsules=True):
    """
    Check all arm pairs at every trajectory step.

    Two-phase evaluation:
      Phase 1  Sphere pre-filter (cheap, conservative):
               Spheres over-approximate links, so if spheres show wide
               clearance (< 0.1*COLLISION_TOL), capsules are guaranteed clear.
      Phase 2  Capsule exact check using corrected frame mapping.

    Returns dict with all fields consumed by kuramoto_sync, including:
      'collision_free_summary'  — human-readable collision status string
      'conflicting_arms'        — for Kuramoto targeted refinement
      'pen_by_pair'             — per-pair max penetration (refinement trigger)
      'time_offsets'            — phase offsets for Kuramoto initialisation
    """
    arm_ids = list(trajectories.keys())
    pairs   = list(combinations(arm_ids, 2))

    mode = 'capsule (URDF-derived)' if use_capsules else 'sphere (legacy)'
    print(f'\n  Arms     : {arm_ids}')
    print(f'  Pairs    : {[(a,b) for a,b in pairs]}')
    print(f'  Mode     : {mode}  [{N_CAP} capsules/arm]')
    print(f'  Geometry : {"URDF-parsed" if CAPSULES is not M1013_CAPSULES else "hand-tuned (M1013 DH-derived)"}')

    n_steps  = min(len(trajectories[a]['trajectory_points']) for a in arm_ids)
    duration = trajectories[arm_ids[0]]['trajectory_points'][-1]['time']
    t_grid   = np.array([
        trajectories[arm_ids[0]]['trajectory_points'][i]['time']
        for i in range(n_steps)
    ])

    joint_arr = {
        aid: np.array([p['joints']
                       for p in trajectories[aid]['trajectory_points'][:n_steps]])
        for aid in arm_ids
    }

    clr_profiles  = {f'{a}_{b}': [] for a, b in pairs}
    max_pen_pair   = {f'{a}_{b}': 0.0 for a, b in pairs}
    min_clr_pair   = {f'{a}_{b}': np.inf for a, b in pairs}
    safe            = True
    max_pen         = 0.0
    first_coll_time = None
    events          = []
    conflict_set: Set[str] = set()
    warn_pairs:   Set[str] = set()
    phase2_calls  = 0  

    for i in range(0, n_steps, EVAL_EVERY_N):
        t = float(t_grid[i])

       
        sph_c = {
            aid: _centres_spheres(joint_arr[aid][i], arm_bases[aid])
            for aid in arm_ids
        }

        
        for aid in arm_ids:
            sp = _self_pen_caps(joint_arr[aid][i]) if use_capsules else 0.0
            if sp > COLLISION_TOL:
                safe    = False
                max_pen = max(max_pen, sp)
                if first_coll_time is None: first_coll_time = t
                conflict_set.add(aid)
                events.append({'time': t, 'type': 'self',
                               'arm': aid, 'penetration': float(sp)})

      
        for a, b in pairs:
            key     = f'{a}_{b}'
            sph_pre = _sphere_pen(sph_c[a], sph_c[b])

            
            if use_capsules and sph_pre < COLLISION_TOL * 0.1:
                mn = _sphere_min_clr(sph_c[a], sph_c[b])
                clr_profiles[key].append(mn)
                min_clr_pair[key] = min(min_clr_pair[key], mn)
                continue

           
            if use_capsules:
                caps_a = _caps_world(joint_arr[a][i], arm_bases[a])
                caps_b = _caps_world(joint_arr[b][i], arm_bases[b])
                pen    = _caps_penetration(caps_a, caps_b)
                clr    = _caps_min_clearance(caps_a, caps_b)
                phase2_calls += 1
            else:
                pen = sph_pre
                clr = _sphere_min_clr(sph_c[a], sph_c[b])

            clr_profiles[key].append(float(clr))
            max_pen_pair[key]  = max(max_pen_pair[key], float(pen))
            min_clr_pair[key]  = min(min_clr_pair[key], float(clr))

            if pen > COLLISION_TOL:
                safe    = False
                max_pen = max(max_pen, pen)
                if first_coll_time is None: first_coll_time = t
                conflict_set.add(a); conflict_set.add(b)
                events.append({'time': t, 'type': 'inter_arm',
                               'arm_a': a, 'arm_b': b,
                               'penetration': float(pen),
                               'clearance':   float(clr)})
            elif 0.0 < clr < WARN_MARGIN:
                warn_pairs.add(key)

    
    if events:
        seen, deduped = set(), []
        for ev in events:
            ekey = ev.get('arm') or f"{ev['arm_a']}_{ev['arm_b']}"
            uid  = (ekey, round(ev['time'] / 0.1))
            if uid not in seen:
                seen.add(uid); deduped.append(ev)
        events = deduped[:20]

    
    time_offsets = {aid: 0.0 for aid in arm_ids}
    if not safe:
        for a, b in pairs:
            key     = f'{a}_{b}'
            clr_arr = np.array(clr_profiles[key]) if clr_profiles[key] \
                      else np.array([])
            if len(clr_arr) == 0:
                continue
            min_clr = float(clr_arr.min())
            if min_clr < SAFETY_MARGIN:
                deficit = SAFETY_MARGIN - min_clr
                ci      = int(np.argmin(clr_arr))
                w0, w1  = max(0, ci - 5), min(n_steps - 1, ci + 5)
                spd_a   = float(np.linalg.norm(
                    np.diff(joint_arr[a][w0:w1], axis=0))) + 1e-6
                spd_b   = float(np.linalg.norm(
                    np.diff(joint_arr[b][w0:w1], axis=0))) + 1e-6
                faster  = b if spd_b > spd_a else a
                offset  = min(duration * 0.3,
                               deficit / (spd_a + spd_b) * duration / n_steps)
                time_offsets[faster] = max(time_offsets[faster], offset)
                print(f'  Kuramoto offset [{a}]<->[{b}]  '
                      f'min_clr={min_clr*100:.1f} cm  '
                      f'deficit={deficit*100:.1f} cm  '
                      f'-> delay [{faster}] by {offset:.4f} s')

    
    if safe:
        min_all = min(min_clr_pair.values()) if min_clr_pair else np.inf
        if min_all < WARN_MARGIN:
            summary = (f'COLLISION-FREE  (min clearance={min_all*100:.1f} cm — '
                       f'TIGHT, < {WARN_MARGIN*100:.0f} cm warning threshold)')
        else:
            summary = (f'COLLISION-FREE  (min clearance={min_all*100:.1f} cm, '
                       f'all pairs > {WARN_MARGIN*100:.0f} cm)')
    else:
        summary = (f'COLLISION DETECTED  max_pen={max_pen*1000:.2f} mm  '
                   f't_first={first_coll_time:.3f} s  '
                   f'conflicting={sorted(conflict_set)}')

    
    sep = '=' * 60
    print(f'\n  {sep}')
    print(f'  COLLISION CHECK RESULT')
    print(f'  {sep}')
    print(f'  {summary}')
    print(f'  {"─"*58}')
    print(f'  safe               : {safe}')
    print(f'  max penetration    : {max_pen*1000:.2f} mm')
    print(f'  phase2 calls       : {phase2_calls} / {n_steps} steps '
          f'(capsule checks skipped by pre-filter)')
    if first_coll_time is not None:
        print(f'  first collision    : t = {first_coll_time:.3f} s')
    if warn_pairs:
        print(f'  tight clearance    : {sorted(warn_pairs)}'
              f'  (< {WARN_MARGIN*100:.0f} cm — safe but narrow)')
    print(f'  events (deduped)   : {len(events)}')
    for ev in events[:5]:
        if ev['type'] == 'self':
            print(f'    t={ev["time"]:.2f}  SELF-COL [{ev["arm"]}]'
                  f'  pen={ev["penetration"]*1000:.1f} mm')
        else:
            print(f'    t={ev["time"]:.2f}  INTER-ARM [{ev["arm_a"]}]<->[{ev["arm_b"]}]'
                  f'  pen={ev["penetration"]*1000:.1f} mm'
                  f'  clr={ev["clearance"]*1000:.1f} mm')
    print(f'  conflicting arms   : {sorted(conflict_set)}'
          f'  <- Kuramoto will refine these')
    print(f'  pen_by_pair (mm)   : '
          f'{dict((k, round(v*1000, 2)) for k,v in max_pen_pair.items())}')
    print(f'  min_clr_pair (cm)  : '
          f'{dict((k, round(v*100, 1)) for k,v in min_clr_pair.items())}')
    print(f'  time_offsets       : {time_offsets}'
          f'  <- phase-seed for Kuramoto')
    print(f'  {sep}')

    return {
        'safe':                  safe,
        'collision_free':        safe,        
        'collision_free_summary': summary,     
        'max_penetration_m':     float(max_pen),
        'first_collision_time':  first_coll_time,
        'n_events':              len(events),
        'collision_events':      events,
        'time_offsets':          time_offsets,
        'conflicting_arms':      sorted(conflict_set),
        'pen_by_pair':           max_pen_pair,
        'min_clr_pair_m':        {k: float(v) for k, v in min_clr_pair.items()},
        'arm_ids':               arm_ids,
        'duration':              float(duration),
        'warn_pairs':            sorted(warn_pairs),
        'geometry':              ('capsule_urdf'
                                  if CAPSULES is not M1013_CAPSULES
                                  else 'capsule_hardcoded'),
        'n_capsules_per_arm':    N_CAP,
        'phase2_capsule_calls':  phase2_calls,
    }




def _resolve_bases(arm_ids):
    bases = {}
    for aid in arm_ids:
        if aid in ARM_REGISTRY:
            bases[aid] = ARM_REGISTRY._arms[aid]['base']
        elif hasattr(RobotBases, f'{aid.upper()}_BASE'):
            bases[aid] = getattr(RobotBases, f'{aid.upper()}_BASE')
        else:
            bases[aid] = np.zeros(3)
    return bases


def run(traj_file='trajectories.json',
        output_file='collision_result.json',
        use_capsules=True):
    """
    Load trajectories.json -> check collisions -> write collision_result.json.
    """
    print('\n' + '=' * 80)
    print('COLLISION CHECKER  [M1013 Capsule Geometry — URDF-aware]')
    print('=' * 80)
    geom_src = ('URDF-parsed'
                if CAPSULES is not M1013_CAPSULES
                else 'hand-tuned  (URDF not found — using DH-derived capsules)')
    print(f'  Geometry source  : {geom_src}')
    print(f'  Capsules / arm   : {N_CAP}')
    print(f'  Safety margin    : {SAFETY_MARGIN*100:.0f} cm')
    print(f'  Warn margin      : {WARN_MARGIN*100:.0f} cm')
    print(f'  Collision tol    : {COLLISION_TOL*1000:.0f} mm')

    with open(traj_file) as f:
        data = json.load(f)

    trajs   = data.get('trajectories', data)
    arm_ids = data.get('arm_ids', list(trajs.keys()))
    bases   = _resolve_bases(arm_ids)

    result = check_trajectories(trajs, bases, use_capsules=use_capsules)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n[geometry={result["geometry"]}]')
    print(f'\n>>> RESULT: {result["collision_free_summary"]}')
    print()
    if result['collision_free']:
        print('[OK] All trajectories COLLISION-FREE -> Kuramoto will synchronise phases')
        if result['warn_pairs']:
            print(f'[!!] Tight clearance warning on: {result["warn_pairs"]}')
            print('     Monitor these pairs in Gazebo visualisation')
    else:
        print(f'[!!] COLLISION DETECTED')
        print(f'     max penetration  : {result["max_penetration_m"]*1000:.1f} mm')
        print(f'     conflicting arms : {result["conflicting_arms"]}')
        print('     Kuramoto adaptive refinement will attempt to resolve this')
        print('     by re-seeding conflicting-arm B-spline control points')
    print()
    print('Next -> ros2 run dual_arm_sync kuramoto_sync')
    return result




def main(args=None):
    try:
        import rclpy
        rclpy.init(args=args)
    except Exception:
        pass
    try:
        run()
    except FileNotFoundError:
        print('[!!] trajectories.json not found')
        print('     Run: ros2 run dual_arm_sync trajectory_generation')
    except Exception:
        import traceback
        traceback.print_exc()
    try:
        import rclpy
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()