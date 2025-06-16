#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurtleMove(Node):
    def __init__(self):
        super().__init__('turtle_move')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.timer = self.create_timer(0.5, self.move_turtle)
        self.count = 0

    def move_turtle(self):
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 1.0
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg}')
        self.count += 1
        if self.count > 100:
            rclpy.shutdown()
            self.get_logger().info('Turtle has moved 100 times, shutting down...')
        self.get_logger().info('Turtle is moving...')

def main(args=None):
    rclpy.init(args=args)
    turtle_move = TurtleMove()
    rclpy.spin(turtle_move)
    turtle_move.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()