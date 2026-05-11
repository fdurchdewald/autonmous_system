import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

class TFFilter(Node):
    def __init__(self):
        super().__init__('tf_filter_node')
        
        # Abonnieren der umbenannten Daten vom Bag
        self.sub = self.create_subscription(TFMessage, '/tf_bag', self.callback, 10)
        
        # Veröffentlichen der sauberen Daten auf das echte /tf
        self.pub = self.create_publisher(TFMessage, '/tf', 10)
        
        self.get_logger().info('--- TF-Filter Node gestartet ---')
        self.get_logger().info('Ich filtere "map -> odom" aus /tf_bag heraus.')

    def callback(self, msg):
        filtered_msg = TFMessage()
        
        # Wir gehen jede einzelne Transformation im Datenpaket durch
        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id
            
            # DEBUG: Zeige jede eingehende Transformation
            # self.get_logger().debug(f'Prüfe: {parent} -> {child}')

            # Hier passiert die Filter-Logik
            if parent == 'map' and child == 'odom':
                # Hier haben wir die Korrektur vom Cartographer gefunden!
                self.get_logger().warn(f'>>> FILTER: Link "{parent} -> {child}" BLOCKIERT (Cartographer-Altlast)')
            else:
                # Alles andere (z.B. odom -> base_link) lassen wir durch
                filtered_msg.transforms.append(transform)
                # Optionales Debugging für erlaubte Links (kann sehr viel Text erzeugen):
                # self.get_logger().info(f'Erlaube: {parent} -> {child}')

        # Wenn nach dem Filtern noch Daten übrig sind, senden wir sie weiter
        if filtered_msg.transforms:
            self.pub.publish(filtered_msg)
        else:
            self.get_logger().info('Paket war leer nach Filterung (nur map->odom enthalten).')

def main():
    rclpy.init()
    node = TFFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
