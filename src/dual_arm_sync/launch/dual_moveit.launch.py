#!/usr/bin/env python3

import os
import yaml
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
    OpaqueFunction,
    GroupAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def _load_yaml(package_name: str, relative_path: str) -> dict:
    path = os.path.join(
        get_package_share_directory(package_name), relative_path
    )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def launch_setup(context, *args, **kwargs):

    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)

    # arm 1
    name1  = LaunchConfiguration("name1").perform(context)
    color1 = LaunchConfiguration("color1").perform(context)
    model1 = LaunchConfiguration("model1").perform(context)
    x1     = LaunchConfiguration("x1").perform(context)
    y1     = LaunchConfiguration("y1").perform(context)
    z1     = LaunchConfiguration("z1").perform(context)
    R1     = LaunchConfiguration("R1").perform(context)
    P1     = LaunchConfiguration("P1").perform(context)
    Y1     = LaunchConfiguration("Y1").perform(context)

    # arm 2
    name2  = LaunchConfiguration("name2").perform(context)
    color2 = LaunchConfiguration("color2").perform(context)
    model2 = LaunchConfiguration("model2").perform(context)
    x2     = LaunchConfiguration("x2").perform(context)
    y2     = LaunchConfiguration("y2").perform(context)
    z2     = LaunchConfiguration("z2").perform(context)
    R2     = LaunchConfiguration("R2").perform(context)
    P2     = LaunchConfiguration("P2").perform(context)
    Y2     = LaunchConfiguration("Y2").perform(context)

    desc2_dir    = get_package_share_directory("dsr_description2")
    gazebo2_dir  = get_package_share_directory("dsr_gazebo2")
    ros_gz_dir   = get_package_share_directory("ros_gz_sim")

    moveit_pkg     = f"dsr_moveit_config_{model1}"
    moveit_pkg_dir = get_package_share_directory(moveit_pkg)

    xacro_dir = os.path.join(desc2_dir, "xacro")

    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": " -r -v 3 empty.sdf"}.items(),
    )

    spawn_robot1 = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo2_dir, "launch",
                                 "dsr_spawn_on_gazebo.launch.py")
                ),
                launch_arguments={
                    "use_gazebo":    "true",
                    "name":          name1,
                    "model":         model1,
                    "color":         color1,
                    "x":             x1,
                    "y":             y1,
                    "z":             z1,
                    "R":             R1,
                    "P":             P1,
                    "Y":             Y1,
                    "use_sim_time":  use_sim_time,
                    "remap_tf":      "false",
                }.items(),
            )
        ],
    )

    spawn_robot2 = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo2_dir, "launch",
                                 "dsr_spawn_on_gazebo.launch.py")
                ),
                launch_arguments={
                    "use_gazebo":    "true",
                    "name":          name2,
                    "model":         model2,
                    "color":         color2,
                    "x":             x2,
                    "y":             y2,
                    "z":             z2,
                    "R":             R2,
                    "P":             P2,
                    "Y":             Y2,
                    "use_sim_time":  use_sim_time,
                    "remap_tf":      "false",
                }.items(),
            )
        ],
    )

    
    def make_rsp(name, model, color):
        """One robot_state_publisher namespaced under `name`."""
        return Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace=name,
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time == "true",
                "robot_description": xacro.process_file(
                    os.path.join(xacro_dir, f"{model}.urdf.xacro"),
                    mappings={"color": color, "gripper": "none"},
                ).toxml(),
                "frame_prefix": f"{name}/",
            }],
        )

    rsp1 = make_rsp(name1, model1, color1)
    rsp2 = make_rsp(name2, model2, color2)

    
    static_tf1 = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_dsr01",
        arguments=[
            x1, y1, z1,          
            Y1, P1, R1,          
            "world",
            f"{name1}/world",    
        ],
    )

    static_tf2 = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_dsr02",
        arguments=[
            x2, y2, z2,
            Y2, P2, R2,
            "world",
            f"{name2}/world",    
        ],
    )

    dual_xacro_path = os.path.join(
        moveit_pkg_dir, "config", f"dual_arm_{model1}.urdf.xacro"
    )

    if not os.path.exists(dual_xacro_path):
        raise FileNotFoundError(
            f"\n\n[dual_moveit] Cannot find: {dual_xacro_path}\n"
            "Please create dsr_moveit_config_m1013/config/dual_arm_m1013.urdf.xacro\n"
            "with two prefixed m1013 macros at (0,+0.5,0) and (0,-0.5,0).\n"
        )

    combined_urdf = xacro.process_file(dual_xacro_path).toxml()
    robot_description        = {"robot_description": combined_urdf}

    srdf_path = os.path.join(moveit_pkg_dir, "config", "dsr.srdf")
    robot_description_semantic = {
        "robot_description_semantic": _read_file(srdf_path)
    }

    kinematics_yaml   = _load_yaml(moveit_pkg, "config/kinematics.yaml")
    joint_limits_yaml = _load_yaml(moveit_pkg, "config/joint_limits.yaml")
    ompl_yaml         = _load_yaml(moveit_pkg, "config/ompl_planning.yaml")

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_yaml},
            {"robot_description_planning":   joint_limits_yaml},
            ompl_yaml,
            {
                "publish_planning_scene":          True,
                "publish_geometry_updates":        True,
                "publish_state_updates":           True,
                "publish_transforms_updates":      True,
                "publish_planning_scene_hz":       10.0,

                "default_planning_pipeline":       "ompl",
                "planning_pipelines":              ["ompl"],

               "use_sim_time":  use_sim_time == "true",
            },
        ],
    )

    rviz_config = os.path.join(moveit_pkg_dir, "config", "moveit.rviz")

    if not os.path.exists(rviz_config):
        rviz_config = os.path.join(desc2_dir, "rviz", "default.rviz")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"use_sim_time": use_sim_time == "true"},
        ],
    )

    fcl_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="dual_arm_sync",        
                executable="dual_arm_fcl_collision",
                name="dual_arm_fcl_collision",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    {
                        "use_sim_time":  use_sim_time == "true",
                        "arm1_prefix":   f"{name1}_",   
                        "arm2_prefix":   f"{name2}_",   
                        "check_rate_hz": 10.0,
                    },
                ],
            )
        ],
    )

    return [
        gazebo,
        spawn_robot1,    
        spawn_robot2,    
        static_tf1,
        static_tf2,
        rsp1,
        rsp2,

        move_group_node,
        rviz_node,      

        fcl_node,        
    ]


def generate_launch_description():

    ARGUMENTS = [
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use Gazebo simulation clock"),

        DeclareLaunchArgument("name1",  default_value="dsr01"),
        DeclareLaunchArgument("color1", default_value="white"),
        DeclareLaunchArgument("model1", default_value="m1013"),
        DeclareLaunchArgument("x1",     default_value="0.0"),
        DeclareLaunchArgument("y1",     default_value="0.5",
            description="Arm-1 Y offset  (+0.5 m from centre)"),
        DeclareLaunchArgument("z1",     default_value="0.0"),
        DeclareLaunchArgument("R1",     default_value="0.0"),
        DeclareLaunchArgument("P1",     default_value="0.0"),
        DeclareLaunchArgument("Y1",     default_value="0.0"),

        DeclareLaunchArgument("name2",  default_value="dsr02"),
        DeclareLaunchArgument("color2", default_value="blue"),
        DeclareLaunchArgument("model2", default_value="m1013"),
        DeclareLaunchArgument("x2",     default_value="0.0"),
        DeclareLaunchArgument("y2",     default_value="-0.5",
            description="Arm-2 Y offset  (-0.5 m from centre)"),
        DeclareLaunchArgument("z2",     default_value="0.0"),
        DeclareLaunchArgument("R2",     default_value="0.0"),
        DeclareLaunchArgument("P2",     default_value="0.0"),
        DeclareLaunchArgument("Y2",     default_value="0.0"),
    ]

    return LaunchDescription(
        ARGUMENTS + [OpaqueFunction(function=launch_setup)]
    )