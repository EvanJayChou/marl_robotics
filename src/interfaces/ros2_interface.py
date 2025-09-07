"""
ROS 2 interface for real robot control and simulation bridging.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from typing import Dict, List, Optional
import numpy as np
import threading

class ROS2Interface(Node):
    def __init__(self, config: dict):
        """
        Initialize ROS 2 interface for robot control.
        
        Args:
            config (dict): Configuration for ROS 2 interface
        """
        super().__init__('bipedal_robot_interface')
        self.config = config
        
        # Initialize publishers and subscribers
        self._setup_communications()
        
        # Initialize transform broadcaster
        from tf2_ros import TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # State variables
        self.joint_states = {}
        self.robot_poses = {}
        self.imu_data = {}
        
        # Threading lock
        self._lock = threading.Lock()
        
    def _setup_communications(self):
        """Setup ROS 2 publishers and subscribers."""
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Publishers
        self.joint_cmd_pub = {}
        self.cmd_vel_pub = {}
        
        # Create publishers for each robot
        for robot_name in self.config["robots"]:
            self.joint_cmd_pub[robot_name] = self.create_publisher(
                Float64MultiArray,
                f'/{robot_name}/joint_commands',
                qos
            )
            
            self.cmd_vel_pub[robot_name] = self.create_publisher(
                Twist,
                f'/{robot_name}/cmd_vel',
                qos
            )
            
        # Subscribers
        self.joint_state_sub = {}
        self.robot_pose_sub = {}
        self.imu_sub = {}
        
        # Create subscribers for each robot
        for robot_name in self.config["robots"]:
            self.joint_state_sub[robot_name] = self.create_subscription(
                JointState,
                f'/{robot_name}/joint_states',
                lambda msg, name=robot_name: self._joint_state_callback(msg, name),
                qos
            )
            
            self.robot_pose_sub[robot_name] = self.create_subscription(
                Odometry,
                f'/{robot_name}/odom',
                lambda msg, name=robot_name: self._odom_callback(msg, name),
                qos
            )
            
            self.imu_sub[robot_name] = self.create_subscription(
                Imu,
                f'/{robot_name}/imu',
                lambda msg, name=robot_name: self._imu_callback(msg, name),
                qos
            )
            
    def send_joint_commands(
        self,
        robot_name: str,
        joint_positions: List[float]
    ):
        """
        Send joint position commands to robot.
        
        Args:
            robot_name: Name of the robot
            joint_positions: List of joint positions
        """
        if robot_name not in self.joint_cmd_pub:
            self.get_logger().error(f"Unknown robot: {robot_name}")
            return
            
        msg = Float64MultiArray()
        msg.data = joint_positions
        self.joint_cmd_pub[robot_name].publish(msg)
        
    def send_velocity_command(
        self,
        robot_name: str,
        linear_vel: List[float],
        angular_vel: List[float]
    ):
        """
        Send velocity commands to robot.
        
        Args:
            robot_name: Name of the robot
            linear_vel: [x, y, z] linear velocity
            angular_vel: [roll, pitch, yaw] angular velocity
        """
        if robot_name not in self.cmd_vel_pub:
            self.get_logger().error(f"Unknown robot: {robot_name}")
            return
            
        msg = Twist()
        msg.linear.x = linear_vel[0]
        msg.linear.y = linear_vel[1]
        msg.linear.z = linear_vel[2]
        msg.angular.x = angular_vel[0]
        msg.angular.y = angular_vel[1]
        msg.angular.z = angular_vel[2]
        
        self.cmd_vel_pub[robot_name].publish(msg)
        
    def get_joint_states(self, robot_name: str) -> Optional[Dict]:
        """
        Get latest joint states for robot.
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            Dict containing joint positions and velocities
        """
        with self._lock:
            return self.joint_states.get(robot_name)
            
    def get_robot_pose(self, robot_name: str) -> Optional[Dict]:
        """
        Get latest robot pose.
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            Dict containing position and orientation
        """
        with self._lock:
            return self.robot_poses.get(robot_name)
            
    def get_imu_data(self, robot_name: str) -> Optional[Dict]:
        """
        Get latest IMU data.
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            Dict containing IMU measurements
        """
        with self._lock:
            return self.imu_data.get(robot_name)
            
    def _joint_state_callback(self, msg: JointState, robot_name: str):
        """Handle joint state messages."""
        with self._lock:
            self.joint_states[robot_name] = {
                'position': list(msg.position),
                'velocity': list(msg.velocity),
                'effort': list(msg.effort),
                'names': list(msg.name)
            }
            
    def _odom_callback(self, msg: Odometry, robot_name: str):
        """Handle odometry messages."""
        with self._lock:
            self.robot_poses[robot_name] = {
                'position': [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ],
                'orientation': [
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z
                ],
                'linear_velocity': [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ],
                'angular_velocity': [
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z
                ]
            }
            
            # Broadcast transform
            self._broadcast_transform(msg.pose.pose, robot_name)
            
    def _imu_callback(self, msg: Imu, robot_name: str):
        """Handle IMU messages."""
        with self._lock:
            self.imu_data[robot_name] = {
                'orientation': [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z
                ],
                'angular_velocity': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ],
                'linear_acceleration': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ]
            }
            
    def _broadcast_transform(self, pose: Pose, robot_name: str):
        """Broadcast robot transform."""
        from geometry_msgs.msg import TransformStamped
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = f'{robot_name}/base_link'
        
        # Copy translation
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        
        # Copy rotation
        t.transform.rotation = pose.orientation
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)
        
    def spin_once(self):
        """Process callbacks once."""
        rclpy.spin_once(self, timeout_sec=0)
        
    def run(self):
        """Run the node."""
        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            pass
        finally:
            self.destroy_node()
            rclpy.shutdown()
