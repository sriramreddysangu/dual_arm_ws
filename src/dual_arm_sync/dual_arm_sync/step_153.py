#!/usr/bin/env python3

import os, sys
os.environ['DUAL_ARM_CELL'] = 'dual'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cell_bases as _CB
import _robot8x as _R8X
_R8X.ARM_NAMES, _R8X.ROBOT_BASES = _CB.ARM_NAMES, _CB.ROBOT_BASES

import json
import numpy as np
from _robot8x import (ROBOT_BASES, ARM_NAMES, pair_min_dist,
                      CLEAR_M, PREFILTER_M, _caps_surface)

def main():
    with open('s152_trajectories.json') as fh: t = json.load(fh)
    arm_names = t.get('arm_names', ARM_NAMES)
    pos = {n: np.array(t[n]['trajectory']['positions'], float) for n in arm_names}
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    N = len(arm_names); results = []
    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = arm_names[i], arm_names[j]
            K = min(len(pos[ni]), len(pos[nj])); arc = np.linspace(0., 1., K)
            first = -1; n_coll = 0; dmin = float('inf')
            for k in range(K):
                if _caps_surface(pos[ni][k], bases[ni],
                                 pos[nj][k], bases[nj]) > PREFILTER_M:
                    continue
                d = pair_min_dist(pos[ni][k], bases[ni], pos[nj][k], bases[nj])
                if d < dmin: dmin = d
                if d < CLEAR_M:
                    n_coll += 1
                    if first < 0: first = k
            results.append({'pair': f'{ni}<->{nj}', 'arm_i': ni, 'arm_j': nj,
                            'status': 'COLLISION' if n_coll > 0 else 'SAFE',
                            'steps_checked': K, 'coll_steps': n_coll,
                            'global_min_dist_m': round(float(min(dmin, 1.0)), 4),
                            'first_collision_arc': round(float(arc[first]), 5)
                            if first >= 0 else None,
                            'per_timestep_detail': []})
    n_coll = sum(1 for r in results if r['status'] == 'COLLISION')
    out = {'overall_status': 'COLLISION' if n_coll else 'SAFE',
           'n_pairs': len(results), 'n_colliding_pairs': n_coll, 'pairs': results}
    with open('s153_collision_map.json', 'w') as fh: json.dump(out, fh)
    print(f'  STEP 153: {out["overall_status"]}, {n_coll}/{len(results)} pairs collide')


if __name__ == '__main__':
    main()