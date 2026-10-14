#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List, Dict

try:
    from dual_arm_sync.constants import DHParameters
    from dual_arm_sync.ik_solver import RobotBases
except ImportError as e:
    print(f"ERROR: Cannot import required modules: {e}")
    import sys
    sys.exit(1)

class LinkGeometry:
    """
    Link geometry as capsules (cylinder + sphere caps)
    Based on actual M1013 collision mesh dimensions
    """
    
    
    CAPSULES = {
        'base_link': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.15]),
            'radius': 0.12
        },
        'link_1': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.12]),
            'radius': 0.10
        },
        'link_2': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.62]),
            'radius': 0.09
        },
        'link_3': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.56]),
            'radius': 0.08
        },
        'link_4': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.10]),
            'radius': 0.06
        },
        'link_5': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.08]),
            'radius': 0.05
        },
        'link_6': {
            'start': np.array([0.0, 0.0, 0.0]),
            'end': np.array([0.0, 0.0, 0.06]),
            'radius': 0.04
        },
    }

def point_to_line_segment_distance(point: np.ndarray,
                                   line_start: np.ndarray,
                                   line_end: np.ndarray) -> float:
    """Compute minimum distance from point to line segment"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq < 1e-10:  
        return np.linalg.norm(point_vec)
    
    t = np.clip(np.dot(point_vec, line_vec) / line_len_sq, 0.0, 1.0)
    projection = line_start + t * line_vec
    
    return np.linalg.norm(point - projection)


def capsule_capsule_distance(cap1_start: np.ndarray, cap1_end: np.ndarray, cap1_radius: float,
                             cap2_start: np.ndarray, cap2_end: np.ndarray, cap2_radius: float) -> float:
    
    
    
    d1 = cap1_end - cap1_start
    
    d2 = cap2_end - cap2_start
    
    r = cap1_start - cap2_start
    
    a = np.dot(d1, d1)  
    e = np.dot(d2, d2)  
    f = np.dot(d2, r)
    
    
    if a <= 1e-10 and e <= 1e-10:
        dist = np.linalg.norm(cap1_start - cap2_start)
        return dist - (cap1_radius + cap2_radius)
    
    if a <= 1e-10:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= 1e-10:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            
            t = (b * s + f) / e
            
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    
    closest1 = cap1_start + s * d1
    closest2 = cap2_start + t * d2
    
    dist = np.linalg.norm(closest1 - closest2)
    
    return dist - (cap1_radius + cap2_radius)

class MeshCollisionChecker:
    
    def __init__(self, safety_margin: float = 0.02):
        
        self.safety_margin = safety_margin
        
        self.self_collision_pairs = [
            (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),  
            (1, 3), (1, 4), (1, 5), (1, 6),         
            (2, 4), (2, 5), (2, 6),                  
            (3, 5), (3, 6),                          
            (4, 6),                                  
        ]
    
    def compute_link_transforms(self, joint_angles: np.ndarray,
                                robot_base: np.ndarray) -> List[np.ndarray]:
        
        dh_params = DHParameters.get_dh_params(joint_angles)
        
        transforms = []
        T = np.eye(4)
        
        for i in range(7):  
            if i == 0:
                
                T_base = np.eye(4)
                T_base[:3, 3] = robot_base
                transforms.append(T_base)
            else:
                
                alpha, a, theta, d = dh_params[i-1]
                
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                cos_alpha = np.cos(alpha)
                sin_alpha = np.sin(alpha)
                
                T_i = np.array([
                    [cos_theta, -sin_theta, 0, a],
                    [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -sin_alpha * d],
                    [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, cos_alpha * d],
                    [0, 0, 0, 1]
                ])
                
                T = T @ T_i
                T_world = np.eye(4)
                T_world[:3, :3] = T[:3, :3]
                T_world[:3, 3] = T[:3, 3] + robot_base
                transforms.append(T_world)
        
        return transforms
    
    def transform_capsule(self, capsule: Dict, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        
        start_homog = np.append(capsule['start'], 1.0)
        end_homog = np.append(capsule['end'], 1.0)
        
        start_world = (transform @ start_homog)[:3]
        end_world = (transform @ end_homog)[:3]
        
        return start_world, end_world, capsule['radius']
    
    def check_self_collision(self, joint_angles: np.ndarray,
                            robot_name: str = 'dsr01') -> Tuple[bool, List, float]:
        
        robot_base = RobotBases.get_base_position(robot_name)
        
        
        transforms = self.compute_link_transforms(joint_angles, robot_base)
        
        
        link_names = ['base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6']
        world_capsules = []
        
        for i, link_name in enumerate(link_names):
            capsule = LinkGeometry.CAPSULES[link_name]
            start, end, radius = self.transform_capsule(capsule, transforms[i])
            world_capsules.append((start, end, radius))
        
        
        collision_pairs = []
        min_distance = float('inf')
        
        for i, j in self.self_collision_pairs:
            start1, end1, radius1 = world_capsules[i]
            start2, end2, radius2 = world_capsules[j]
            
            dist = capsule_capsule_distance(start1, end1, radius1,
                                           start2, end2, radius2)
            
            if dist < min_distance:
                min_distance = dist
            
            if dist < self.safety_margin:
                collision_pairs.append((i, j, dist))
        
        has_collision = len(collision_pairs) > 0
        
        return has_collision, collision_pairs, min_distance
    
    def check_inter_arm_collision(self, joints1: np.ndarray, joints2: np.ndarray) -> Tuple[bool, List, float]:
        
        base1 = RobotBases.DSR01_BASE
        base2 = RobotBases.DSR02_BASE
        
        
        transforms1 = self.compute_link_transforms(joints1, base1)
        transforms2 = self.compute_link_transforms(joints2, base2)
        
        
        link_names = ['base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6']
        
        capsules1 = []
        for i, link_name in enumerate(link_names):
            capsule = LinkGeometry.CAPSULES[link_name]
            start, end, radius = self.transform_capsule(capsule, transforms1[i])
            capsules1.append((start, end, radius))
        
        capsules2 = []
        for i, link_name in enumerate(link_names):
            capsule = LinkGeometry.CAPSULES[link_name]
            start, end, radius = self.transform_capsule(capsule, transforms2[i])
            capsules2.append((start, end, radius))
        
        
        collision_pairs = []
        min_distance = float('inf')
        
        for i in range(len(capsules1)):
            for j in range(len(capsules2)):
                start1, end1, radius1 = capsules1[i]
                start2, end2, radius2 = capsules2[j]
                
                dist = capsule_capsule_distance(start1, end1, radius1,
                                               start2, end2, radius2)
                
                if dist < min_distance:
                    min_distance = dist
                
                if dist < self.safety_margin:
                    collision_pairs.append((i, j, dist))
        
        has_collision = len(collision_pairs) > 0
        
        return has_collision, collision_pairs, min_distance
    
    def check_all_collisions(self, joints1: np.ndarray, joints2: np.ndarray) -> Dict:
        
        self_col1, pairs1, dist1 = self.check_self_collision(joints1, 'dsr01')
        self_col2, pairs2, dist2 = self.check_self_collision(joints2, 'dsr02')
        
        inter_col, pairs_inter, dist_inter = self.check_inter_arm_collision(joints1, joints2)
        
        overall_safe = not (self_col1 or self_col2 or inter_col)
        min_dist_overall = min(dist1, dist2, dist_inter)
        
        details = f"""Collision Check Results:
  DSR01 Self: {'❌ COLLISION' if self_col1 else '✅ Safe'} (min: {dist1*100:.1f}cm)
  DSR02 Self: {'❌ COLLISION' if self_col2 else '✅ Safe'} (min: {dist2*100:.1f}cm)
  Inter-Arm:  {'❌ COLLISION' if inter_col else '✅ Safe'} (min: {dist_inter*100:.1f}cm)
  Overall: {'❌ UNSAFE' if not overall_safe else '✅ SAFE'}
"""
        
        return {
            'self_collision_dsr01': self_col1,
            'self_collision_dsr02': self_col2,
            'inter_collision': inter_col,
            'overall_safe': overall_safe,
            'min_distance': min_dist_overall,
            'details': details,
            'collision_pairs_dsr01': pairs1,
            'collision_pairs_dsr02': pairs2,
            'collision_pairs_inter': pairs_inter,
        }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("MESH-BASED COLLISION CHECKER TEST")
    print("="*80)
    
    checker = MeshCollisionChecker(safety_margin=0.02)
    
    print("\n[Test 1] Home position")
    joints_home = np.zeros(6)
    has_col, pairs, min_dist = checker.check_self_collision(joints_home, 'dsr01')
    print(f"  Self-collision: {has_col}")
    print(f"  Min distance: {min_dist*100:.1f}cm")
    
    print("\n[Test 2] Extreme bend")
    joints_extreme = np.array([0.0, -2.5, -2.0, 0.0, 3.0, 0.0])
    has_col, pairs, min_dist = checker.check_self_collision(joints_extreme, 'dsr01')
    print(f"  Self-collision: {has_col}")
    print(f"  Min distance: {min_dist*100:.1f}cm")
    if pairs:
        print(f"  Collision pairs: {len(pairs)}")
    
    print("\n[Test 3] Dual arm - both at home")
    result = checker.check_all_collisions(joints_home, joints_home)
    print(result['details'])
    
    print("="*80 + "\n")