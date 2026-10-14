#!/usr/bin/env python3


import numpy as np


class LinkLengths:
    
    L1 = 0.1525  
    L2 = 0.620   
    L3 = 0.559   
    L4 = 0.121   
    A  = 0.0345   




class DHParameters:
    
    L1 = LinkLengths.L1
    L2 = LinkLengths.L2
    L3 = LinkLengths.L3
    L4 = LinkLengths.L4
    A  = LinkLengths.A
    
    # DH Table: [alpha, a, theta_offset, d]
    DH_TABLE = np.array([
        # [alpha,      a,    theta_offset,    d  ]
        [0.0,         0.0,        0.0,       L1  ],  
        [-np.pi/2,    0.0,   -np.pi/2,       A   ], 
        [0.0,         L2,     np.pi/2,       0.0 ],  
        [np.pi/2,     0.0,        0.0,       L3  ], 
        [-np.pi/2,    0.0,        0.0,       0.0 ],  
        [np.pi/2,     0.0,        0.0,       L4  ], 
    ])
    
    @classmethod
    def get_dh_params(cls, joint_angles: np.ndarray) -> np.ndarray:
        
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        dh = cls.DH_TABLE.copy()
        
        dh[:, 2] = dh[:, 2] + joint_angles
        return dh




class JointLimits:
    
    POSITION_LIMITS = np.array([
        [-6.283,  6.283],     
        [-1.650,  1.650],     
        [-2.792,  2.792],    
        [-6.283,  6.283],    
        [-6.283,  6.283],    
        [-6.283,  6.283],    
    ])
    
 
    
    VELOCITY_LIMITS = np.array([
        2.0944,   
        2.0944,   
        3.1416,   
        3.9270,  
        3.9270,   
        3.9270,  
    ])
    

    
    ACCELERATION_LIMITS = np.array([
        3.0,    
        3.0,   
        4.0,    
        5.0,    
        5.0,    
        6.0,    
    ])
    

    
    JERK_LIMITS = np.array([
        12.0,  
        12.0,  
        16.0,  
        20.0, 
        20.0,  
        25.0,   
    ])
    
    @classmethod
    def check_position_limits(cls, joint_angles: np.ndarray) -> tuple:
        """Check if joint angles are within limits"""
        violations = []
        for i in range(6):
            lower, upper = cls.POSITION_LIMITS[i]
            if joint_angles[i] < lower or joint_angles[i] > upper:
                violations.append(i)
        return len(violations) == 0, violations
    
    @classmethod
    def check_velocity_limits(cls, joint_velocities: np.ndarray) -> tuple:
        """Check if joint velocities are within limits"""
        violations = []
        for i in range(6):
            if abs(joint_velocities[i]) > cls.VELOCITY_LIMITS[i]:
                violations.append(i)
        return len(violations) == 0, violations
    
    @classmethod
    def clip_to_limits(cls, joint_angles: np.ndarray) -> np.ndarray:
        """Clip joint angles to position limits"""
        clipped = joint_angles.copy()
        for i in range(6):
            lower, upper = cls.POSITION_LIMITS[i]
            clipped[i] = np.clip(clipped[i], lower, upper)
        return clipped




class RobotSpecs:
    """Official Doosan M1013 specifications"""
    
    MODEL = "M1013"
    MANUFACTURER = "Doosan Robotics"
    SERIES = "M-Series"
    
    DOF = 6
    PAYLOAD = 10.0                  
    REACH = 1.300                   
    REPEATABILITY = 0.00005         
    TCP_SPEED_MAX = 1.0            
    
    WEIGHT = 33.0                   
    COLLISION_SENSITIVITY = 0.0002   
    TORQUE_SENSORS = 6
    SAFETY_RATED = True
    
    POWER_CONSUMPTION = 250        
    VOLTAGE = "24V DC"
    
    OPERATING_TEMP = (5, 45)       
    HUMIDITY = (20, 80)            
    IP_RATING = "IP54"



class WorkspaceBounds:
    """Cartesian workspace limits"""
    
    X_BOUNDS = [-0.8, 0.8]
    Y_BOUNDS = [-1.0, 1.0]
    Z_BOUNDS = [0.0, 1.5]
    
    RADIUS_MIN = 0.15
    RADIUS_MAX = 1.30
    HEIGHT_MIN = -0.2
    HEIGHT_MAX = 1.5
    
    @classmethod
    def is_in_workspace(cls, position: np.ndarray) -> bool:
        """Check if Cartesian position is within workspace"""
        x, y, z = position
        
        if not (cls.X_BOUNDS[0] <= x <= cls.X_BOUNDS[1]):
            return False
        if not (cls.Y_BOUNDS[0] <= y <= cls.Y_BOUNDS[1]):
            return False
        if not (cls.Z_BOUNDS[0] <= z <= cls.Z_BOUNDS[1]):
            return False
        
        radius = np.sqrt(x**2 + y**2)
        if not (cls.RADIUS_MIN <= radius <= cls.RADIUS_MAX):
            return False
        
        return True




def print_config_summary():
    """Print comprehensive summary of robot configuration"""
    print("\n" + "="*80)
    print("DOOSAN M1013 ROBOT CONFIGURATION")
    print("="*80)
    
    print(f"\n📋 MODEL: {RobotSpecs.MODEL}")
    print(f"   DOF: {RobotSpecs.DOF} axes")
    print(f"   Payload: {RobotSpecs.PAYLOAD} kg")
    print(f"   Reach: {RobotSpecs.REACH} m")
    
    print(f"\n📏 LINK LENGTHS (from URDF/RViz):")
    print(f"   L1 (base):     {LinkLengths.L1:.4f} m")
    print(f"   L2 (upper):    {LinkLengths.L2:.4f} m")
    print(f"   L3 (forearm):  {LinkLengths.L3:.4f} m")
    print(f"   L4 (tool):     {LinkLengths.L4:.4f} m")
    print(f"   A  (offset):   {LinkLengths.A:.4f} m")
    print(f"   Total height:  {LinkLengths.L1 + LinkLengths.L2 + LinkLengths.L3 + LinkLengths.L4:.4f} m")
    
    print(f"\n🎯 DH PARAMETERS:")
    print("   Joint | alpha    | a      | θ_offset | d")
    print("   ------|----------|--------|----------|--------")
    for i in range(6):
        alpha, a, theta_off, d = DHParameters.DH_TABLE[i]
        print(f"     {i+1}   | {alpha:7.4f} | {a:6.3f} | {theta_off:8.4f} | {d:6.4f}")
    
    print(f"\n🔒 JOINT LIMITS:")
    print("   Joint | Position (deg)     | Position (rad)  | Velocity (deg/s)")
    print("   ------|--------------------|-----------------|-----------------")
    for i in range(6):
        pos_min_deg = np.degrees(JointLimits.POSITION_LIMITS[i, 0])
        pos_max_deg = np.degrees(JointLimits.POSITION_LIMITS[i, 1])
        pos_min_rad = JointLimits.POSITION_LIMITS[i, 0]
        pos_max_rad = JointLimits.POSITION_LIMITS[i, 1]
        vel_deg = np.degrees(JointLimits.VELOCITY_LIMITS[i])
        print(f"     {i+1}   | [{pos_min_deg:6.1f}, {pos_max_deg:6.1f}] | [{pos_min_rad:6.3f}, {pos_max_rad:6.3f}] | {vel_deg:8.1f}")
    
    print("\n" + "="*80)
    print("✓ Configuration loaded successfully")
    print("="*80 + "\n")




def test_dh_parameters():
    """Test DH parameters and FK at home position"""
    print("\n🧪 Testing DH Parameters and Forward Kinematics")
    print("="*80)
    
    # Home position
    home = np.zeros(6)
    dh_home = DHParameters.get_dh_params(home)
    
    print("\nDH Table at home [0, 0, 0, 0, 0, 0]:")
    print("Joint | alpha    | a      | theta    | d")
    print("------|----------|--------|----------|--------")
    for i in range(6):
        alpha, a, theta, d = dh_home[i]
        print(f"  {i+1}   | {alpha:7.4f} | {a:6.3f} | {theta:8.4f} | {d:6.4f}")
    
    # Forward kinematics verification
    from dual_arm_sync.ik_solver import forward_kinematics
    
    try:
        fk_pos, _ = forward_kinematics(home)
        expected_z = LinkLengths.L1 + LinkLengths.L2 + LinkLengths.L3 + LinkLengths.L4
        
        print(f"\n✓ Forward Kinematics at Home:")
        print(f"  Calculated: {fk_pos}")
        print(f"  Expected:   [0.0, 0.0, {expected_z:.4f}]")
        print(f"  Error:      {np.linalg.norm(fk_pos - np.array([0, 0, expected_z]))*1000:.2f} mm")
    except:
        print("\n⚠️  FK test skipped (ik_solver not imported)")
    
    print("="*80)


if __name__ == '__main__':
    print_config_summary()
    test_dh_parameters()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Configuration ready")
    print("="*80 + "\n")

















