#!/usr/bin/env python3
"""
tf_filter.py  —  Leitet /tf_bag → /tf weiter, OHNE den map→odom Transform.

Warum: Der Bag enthaelt Cartographers map→odom Korrekturen in /tf.
       Beim Abspielen wuerden diese mit unserem EKF-map→odom kollidieren.
       Loesung: Bag mit --remap /tf:=/tf_bag abspielen, dieser Node
       filtert map→odom heraus und republisht alles andere auf /tf.

Starten (in eigenem Terminal):
  python3 tf_filter.py
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class TFFilter(Node):

    def __init__(self):
        super().__init__('tf_filter')
        self.pub = self.create_publisher(TFMessage, '/tf', 100)
        self.create_subscription(TFMessage, '/tf_bag', self._cb, 100)
        self.get_logger().info(
            'TF-Filter aktiv:  /tf_bag → /tf  (filtert map→odom heraus)')

    def _cb(self, msg: TFMessage):
        out = TFMessage()
        out.transforms = [
            t for t in msg.transforms
            if not (t.header.frame_id == 'map' and t.child_frame_id == 'odom')
        ]
        if out.transforms:
            self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(TFFilter())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
