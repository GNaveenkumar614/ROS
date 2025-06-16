#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class DrawPentagon(Node):
    def __init__(self):
        super().__init__('draw_pentagon')
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Pentagon parameters (tuned for turtlesim)
        self.side_length = 2.0       # Virtual "meters"
        self.linear_speed = 1.0      # m/s
        self.angular_speed = 1.0     # rad/s
        self.turn_angle = math.pi * 2 / 5  # 72° in radians (360°/5 sides)
        
        # Calculate timings
        self.side_duration = self.side_length / self.linear_speed
        self.turn_duration = self.turn_angle / self.angular_speed
        
        # State tracking
        self.state = 'forward'
        self.start_time = self.get_clock().now()
        self.sides_completed = 0
        self.pentagons_completed = 0
        
        # Control at 30Hz
        self.timer = self.create_timer(0.033, self.update)

    def update(self):
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        
        msg = Twist()
        
        if self.state == 'forward':
            msg.linear.x = self.linear_speed
            if elapsed >= self.side_duration:
                self.state = 'turning'
                self.start_time = current_time
                self.sides_completed += 1
                self.get_logger().info(f'Completed side {self.sides_completed}')
        
        elif self.state == 'turning':
            msg.angular.z = self.angular_speed
            if elapsed >= self.turn_duration:
                if self.sides_completed >= 5:
                    self.pentagons_completed += 1
                    self.sides_completed = 0
                    self.get_logger().info(f'Completed pentagon {self.pentagons_completed}')
                self.state = 'forward'
                self.start_time = current_time
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DrawPentagon()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\nStopping pentagon drawing...')
    finally:
        # Stop the turtle
        stop_msg = Twist()
        node.publisher.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()