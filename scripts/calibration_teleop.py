"""Slow, deadman-gated joystick control for manual wrist calibration poses.

No gripper control or standard-pose shortcuts. Button 4 selects XY tilts.
Run only with the policy/control engine and normal teleop mapper stopped.
"""
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped


def joystick_twist(axes, buttons, enabled):
    axis = lambda i: max(-1., min(1., float(axes[i]))) if i < len(axes) else 0.
    button = lambda i: bool(buttons[i]) if i < len(buttons) else False
    linear, angular = [0., 0., 0.], [0., 0., 0.]
    if any(not math.isfinite(float(value)) for value in axes):
        return linear, angular
    if not enabled or not button(0):
        return linear, angular
    if button(3):
        angular = [-axis(1) * .10, axis(0) * .10, axis(2) * .10]
    else:
        linear = [-axis(1) * .015, axis(0) * .015, (int(button(5)) - int(button(4))) * .015]
        angular[2] = axis(2) * .10
    return linear, angular


class CalibrationTeleop(Node):
    def __init__(self):
        super().__init__('wrist_calibration_teleop')
        self.publisher = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.receive, qos_profile_sensor_data)
        self.axes, self.buttons, self.received = [], [], 0.
        self.release_seen = False
        self.conflict = False
        self.timer = self.create_timer(.02, self.update)
        self.get_logger().info('CALIBRATION ONLY: release trigger first. Trigger: 15 mm/s translation; trigger + physical button 4: 0.10 rad/s XY tilt. Gripper/reset buttons disabled.')

    def receive(self, msg):
        self.axes, self.buttons, self.received = list(msg.axes), list(msg.buttons), time.monotonic()
        if self.buttons and not self.buttons[0]:
            self.release_seen = True

    def update(self):
        nodes = self.get_node_names()
        conflict = ('control_engine' in nodes or 'logitech_servo_mapper' in nodes
                    or self.count_publishers('/control_abstraction_layer') > 0
                    or self.count_publishers('/servo_node/delta_twist_cmds') > 1)
        if conflict and not self.conflict:
            self.get_logger().error('Competing command source: calibration joystick latched OFF; inspect and restart')
            self.conflict = True
        enabled = self.release_seen and not self.conflict and time.monotonic() - self.received < .25
        linear, angular = joystick_twist(self.axes, self.buttons, enabled)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = linear
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = angular
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = CalibrationTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.header.frame_id = 'base_link'
        node.publisher.publish(stop)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
