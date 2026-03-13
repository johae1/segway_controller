#!/usr/bin/env python3
"""
State Vector Node für Balancierroboter
=======================================
Subscriber:  /imu, /odom
Publisher:   /state_vector  (Float64MultiArray)
             Format: [s_dot_raw, s_dot_filtered, phi, phi_dot]

Starten:
  ros2 run <package> state_vector_node
Aufzeichnen:
  ros2 bag record /state_vector
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


def quaternion_to_phi(x, y, z, w) -> float:
    """Quaternion -> Nickwinkel phi (Pitch)."""
    sin_pitch = 2.0 * (w * y - z * x)
    return np.arcsin(float(np.clip(sin_pitch, -1.0, 1.0)))


class LowPassFilter:
    """Diskreter Tiefpass 1. Ordnung (Tustin)."""

    def __init__(self, f_grenz: float, Ts: float):
        omega_c = 2.0 * np.pi * f_grenz
        k       = omega_c * Ts
        self.b0 =  k / (2.0 + k)
        self.b1 =  k / (2.0 + k)
        self.a1 = -(2.0 - k) / (2.0 + k)
        self.x_prev = 0.0
        self.y_prev = 0.0

    def update(self, x: float) -> float:
        y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev
        self.x_prev = x
        self.y_prev = y
        return y


class StateVectorNode(Node):
    def __init__(self):
        super().__init__('state_vector_node')

        # Parameter
        self.Ts      = 0.005   # Abtastzeit [s]
        self.f_grenz = 10.0    # Tiefpass-Grenzfrequenz [Hz]

        # Zustand
        self.s_dot          = 0.0
        self.s_dot_filtered = 0.0
        self.phi            = 0.0
        self.phi_dot        = 0.0
        self.imu_received   = False
        self.odom_received  = False

        # Tiefpass
        self.lp_filter = LowPassFilter(self.f_grenz, self.Ts)

        # Subscriber
        self.create_subscription(Imu,      '/imu',  self.imu_callback,  10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher
        self.state_pub = self.create_publisher(Float64MultiArray, '/state_vector', 10)

        # Timer: 200 Hz
        self.create_timer(self.Ts, self.publish_state)

        self.get_logger().info('State Vector Node gestartet.')

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.phi          = quaternion_to_phi(q.x, q.y, q.z, q.w)
        self.phi_dot      = msg.angular_velocity.y
        self.imu_received = True

    def odom_callback(self, msg: Odometry):
        self.s_dot          = msg.twist.twist.linear.x
        self.s_dot_filtered = self.lp_filter.update(self.s_dot)
        self.odom_received  = True

    def publish_state(self):
        if not (self.imu_received and self.odom_received):
            return

        msg = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label  = 's_dot_raw,s_dot_filtered,phi,phi_dot'
        dim.size   = 4
        dim.stride = 4
        msg.layout.dim = [dim]
        msg.data = [
            self.s_dot,
            self.s_dot_filtered,
            self.phi,
            self.phi_dot,
        ]
        self.state_pub.publish(msg)

    def get_state(self) -> np.ndarray:
        """Gefilterter Zustandsvektor: [s_dot, phi, phi_dot]"""
        return np.array([self.s_dot_filtered, self.phi, self.phi_dot])


def main(args=None):
    rclpy.init(args=args)
    node = StateVectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()