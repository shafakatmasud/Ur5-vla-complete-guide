"""Manual two-apple demonstrations; no policy or automatic arm repositioning."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    config=Path(__file__).resolve().parents[1]/'config/recorder_experiment2_sort_fruits.yaml'
    root=Path(__file__).resolve().parents[4]
    return LaunchDescription([
        ExecuteProcess(cmd=['/usr/bin/python3',str(root/'scripts/demonstration_monitor.py'),
                            '--experiment-root',str(root/'data/dataset/staging/experiment_2_sort_fruits')],
                       output='screen'),
        Node(package='ur5_vla_dataset_recorder',executable='logitech_servo',
             name='logitech_servo_mapper',output='screen'),
        Node(package='ur5_vla_dataset_recorder',executable='dataset_recorder',
             name='dataset_recorder',output='screen',parameters=[str(config)]),
    ])
