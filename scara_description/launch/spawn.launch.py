# Copyright (c) 2025 ROBOTECH ASSOCIATION
# Autored by Justo Dario Valverde Torres
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # Initial robot pose
    declare_x_cmd = DeclareLaunchArgument('x', default_value='0.0')
    declare_y_cmd = DeclareLaunchArgument('y', default_value='0.0')
    declare_z_cmd = DeclareLaunchArgument('z', default_value='2.70')
    declare_roll_cmd = DeclareLaunchArgument('R', default_value='0.0')
    declare_pitch_cmd = DeclareLaunchArgument('P', default_value='0.0')
    declare_yaw_cmd = DeclareLaunchArgument('Y', default_value='0.0')
    lidar_arg = DeclareLaunchArgument(
        'lidar',
        default_value='false',
        description='Enable lidar sensor'
    )

    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='false',
        description='Enable camera sensor'
    )
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    name_arg = DeclareLaunchArgument(
        'name',
        default_value='scara',
        description='Model name used in gazebo',
    )
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace to apply to the nodes, topics and TF frames'
    )
    declare_do_tf_remapping_arg = DeclareLaunchArgument(
        'do_tf_remapping',
        default_value='False',
        description='Whether to remap the tf topic to independent namespaces (/tf -> tf)',
    )

    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('scara_description'),
            'launch/'), 'scara_description.launch.py']),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'do_tf_remapping': LaunchConfiguration('do_tf_remapping'),
            'gazebo': 'true',
            'camera': LaunchConfiguration('camera'),
            'lidar': LaunchConfiguration('lidar'),
        }.items()
    )

    gazebo_spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        namespace = LaunchConfiguration('namespace'),
        arguments=[
            "-name",
            LaunchConfiguration('name'),
            "-topic",
            "robot_description",
            "-x", LaunchConfiguration('x'),
            "-y", LaunchConfiguration('y'),
            "-z", LaunchConfiguration('z'),
            "-R", LaunchConfiguration('R'),
            "-P", LaunchConfiguration('P'),
            "-Y", LaunchConfiguration('Y'),
        ],
    )
    
    ld = LaunchDescription()
    ld.add_action(declare_x_cmd)
    ld.add_action(declare_y_cmd)
    ld.add_action(declare_z_cmd)
    ld.add_action(declare_roll_cmd)
    ld.add_action(declare_pitch_cmd)
    ld.add_action(declare_yaw_cmd)
    ld.add_action(camera_arg)
    ld.add_action(lidar_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(name_arg)
    ld.add_action(namespace_arg)
    ld.add_action(declare_do_tf_remapping_arg)
    ld.add_action(robot_description)
    ld.add_action(gazebo_spawn_robot)

    return ld
