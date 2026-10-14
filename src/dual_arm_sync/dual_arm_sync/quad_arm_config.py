#!/usr/bin/env python3

import numpy as np

N_ARMS = 4

ARM_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']

ARM_COLORS = ['white', 'blue', 'blue', 'white']

BASE_POSITIONS = {
    'dsr01': np.array([ 0.0,  0.5, 0.0]),
    'dsr02': np.array([ 0.0, -0.5, 0.0]),
    'dsr03': np.array([ 1.0,  0.5, 0.0]),
    'dsr04': np.array([ 1.0, -0.5, 0.0]),
}

MODEL = 'm1013'

MIN_SAFE_DISTANCE   = 0.15   
SAFETY_MARGIN       = 0.15   
LINK_RADII          = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])

JOINT_POSITION_LIMITS = np.array([
    [-6.283,  6.283],  
    [-1.650,  1.650],   
    [-2.792,  2.792],  
    [-6.283,  6.283],   
    [-6.283,  6.283],  
    [-6.283,  6.283],  
])

JOINT_VELOCITY_LIMITS = np.array([
    2.0944, 2.0944, 3.1416, 3.9270, 3.9270, 3.9270
])

JOINT_ACCELERATION_LIMITS = np.array([3.0, 3.0, 4.0, 5.0, 5.0, 6.0])

WORKSPACE = {
    'x': (-0.8,  0.8),
    'y': (-1.0,  1.0),
    'z': ( 0.0,  1.5),
    'radius_min': 0.15,
    'radius_max': 1.30,
}

COLLISION_PAIRS = [
    ('dsr01', 'dsr02'),
    ('dsr01', 'dsr03'),
    ('dsr01', 'dsr04'),
    ('dsr02', 'dsr03'),
    ('dsr02', 'dsr04'),
    ('dsr03', 'dsr04'),
]


def get_base(name: str) -> np.ndarray:
    
    return BASE_POSITIONS[name].copy()


def arm_index(name: str) -> int:
    
    return ARM_NAMES.index(name)


def print_config():
    print(f"\n{'='*60}")
    print(f"  QUAD-ARM CONFIG  (N={N_ARMS})")
    print(f"{'='*60}")
    for name in ARM_NAMES:
        b = BASE_POSITIONS[name]
        print(f"  {name}  pos=({b[0]:+.1f},{b[1]:+.1f},{b[2]:.1f})"
              f"  color={ARM_COLORS[arm_index(name)]}")
    print(f"\n  Collision pairs: {len(COLLISION_PAIRS)}")
    for p in COLLISION_PAIRS:
        print(f"    {p[0]} <-> {p[1]}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    print_config()