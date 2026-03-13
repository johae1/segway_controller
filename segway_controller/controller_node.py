#!/usr/bin/env python3
"""
Controller Node für Balancierroboter
======================================
Subscriber:  /state_vector  (Float64MultiArray)
Publisher:   /cmd_vel       (Twist)

Regler:  diskreter LQR mit Integrator auf s_dot
         x = [s_dot, phi, phi_dot, x_err]
         u = -K_ext @ x

Sicherheit:
  - Regler nur aktiv wenn |phi| < phi_max
  - Stellgröße begrenzt auf [-u_max, u_max]

Starten:
  ros2 run <package> controller_node
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        # ---------------------------------------------------------------
        # Regler-Parameter (aus MATLAB übernehmen!)
        # K_ext = dlqr(Ad_ext, Bd_ext, Q_ext, R_ext)
        # x     = [s_dot, phi, phi_dot, x_err]
        # ---------------------------------------------------------------
        self.K_ext = np.array([-1.8755, -29.2582, -0.9120, 0.4696])

        # Abtastzeit
        self.Ts = 0.005   # [s]

        # Sicherheitsgrenzen
        self.phi_max = np.deg2rad(8.0)   # max. Neigungswinkel [rad]
        self.u_max   = 0.2               # max. Sollgeschwindigkeit [m/s]

        # Integrator-Zustand
        self.x_err     = 0.0
        self.x_err_max = 0.5   # Anti-Windup

        # Sollwert
        self.s_dot_ref = 0.0   # [m/s]

        # Zustand
        self.s_dot          = 0.0
        self.phi            = 0.0
        self.phi_dot        = 0.0
        self.state_received = False
        self.active         = False

        # Subscriber
        self.create_subscription(
            Float64MultiArray, '/state_vector', self.state_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Regler-Timer: 200 Hz
        self.create_timer(self.Ts, self.control_loop)

        self.get_logger().info(
            f'Controller Node gestartet.\n'
            f'  K_ext   = {self.K_ext}\n'
            f'  phi_max = {np.degrees(self.phi_max):.1f}°\n'
            f'  u_max   = {self.u_max} m/s'
        )

    # ------------------------------------------------------------------
    def state_callback(self, msg: Float64MultiArray):
        # Format: [s_dot_raw, s_dot_filtered, phi, phi_dot]
        self.s_dot          = msg.data[1]   # gefiltert
        self.phi            = msg.data[2]
        self.phi_dot        = msg.data[3]
        self.state_received = True

    # ------------------------------------------------------------------
    def control_loop(self):
        if not self.state_received:
            return

        # Sicherheitscheck
        if abs(self.phi) > self.phi_max:
            if self.active:
                self.get_logger().warn(
                    f'phi = {np.degrees(self.phi):.1f}° > {np.degrees(self.phi_max):.1f}° '
                    f'→ Regler deaktiviert!'
                )
            self.active = False
            self.x_err  = 0.0   # Integrator zurücksetzen
            self._publish_cmd(0.0)
            return

        self.active = True

        # Integrator: x_err += (s_dot - s_dot_ref) * Ts
        self.x_err += (self.s_dot - self.s_dot_ref) * self.Ts
        self.x_err  = float(np.clip(self.x_err, -self.x_err_max, self.x_err_max))

        # Zustandsvektor
        x = np.array([
            self.s_dot - self.s_dot_ref,
            self.phi,
            self.phi_dot,
            self.x_err,
        ])

        # Stellgröße
        u = float(-self.K_ext @ x)
        u = float(np.clip(u, -self.u_max, self.u_max))

        self._publish_cmd(u)

    # ------------------------------------------------------------------
    def _publish_cmd(self, v: float):
        msg = Twist()
        msg.linear.x = v
        self.cmd_pub.publish(msg)

    def stop(self):
        self._publish_cmd(0.0)
        self.get_logger().info('Controller gestoppt.')


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()