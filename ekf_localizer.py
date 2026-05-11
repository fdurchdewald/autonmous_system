#!/usr/bin/env python3
"""
ekf_localizer.py  —  EKF-Lokalisierung in einer fertigen Karte (ROS 2 Humble)

Starten:
  python3 ekf_localizer.py --ros-args -p map_yaml:=/pfad/zur/map.yaml

Subscriptions:
  /odom         nav_msgs/Odometry
  /scan         sensor_msgs/LaserScan
  /initialpose  geometry_msgs/PoseWithCovarianceStamped  ← RViz "2D Pose Estimate"

Publications:
  /tf           map → odom  (TF-Broadcaster)
  /ekf_pose     geometry_msgs/PoseStamped
  /map          nav_msgs/OccupancyGrid  (einmalig, latched)

Parameter:
  map_yaml   Pfad zur .yaml Datei der gespeicherten Karte   [PFLICHT]
  n_beams    Anzahl LiDAR-Strahlen fuer den Update-Schritt  [15]
  sigma_qd   Prozessrauschen Distanz [m]                    [0.08]
  sigma_qth  Prozessrauschen Yaw [rad]                      [0.05]
  sigma_r    Messrauschen pro LiDAR-Strahl [m]              [0.15]
  eps_h      Schrittweite numerischer Jacobian              [0.10]
  gate       Outlier-Gate [m]                               [0.50]
  max_range  LiDAR-Maximalreichweite [m]                    [3.50]
"""

import time
import numpy as np
import yaml
from pathlib import Path
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
)

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseWithCovarianceStamped, PoseStamped, TransformStamped
)
import tf2_ros


# ── 2D SE(2) Hilfsfunktionen ──────────────────────────────────────────────────

def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def yaw_to_quat(yaw):
    h = float(yaw) * 0.5
    return 0.0, 0.0, float(np.sin(h)), float(np.cos(h))


def compose(xa, ya, ta, xb, yb, tb):
    """Verkettung zweier 2D-Posen: T_ac = T_ab * T_bc"""
    c, s = np.cos(ta), np.sin(ta)
    return xa + c * xb - s * yb, ya + s * xb + c * yb, ta + tb


def inv_pose(x, y, th):
    """Inverse einer 2D-Pose"""
    c, s = np.cos(th), np.sin(th)
    return -x * c - y * s, x * s - y * c, -th


# ── Node ──────────────────────────────────────────────────────────────────────

class EKFLocalizer(Node):

    def __init__(self):
        super().__init__('ekf_localizer')

        # Parameter ─────────────────────────────────────────────────────────
        self.declare_parameter('map_yaml',  '')
        self.declare_parameter('n_beams',   12)
        self.declare_parameter('sigma_qd',  0.08)
        self.declare_parameter('sigma_qth', 0.05)
        self.declare_parameter('sigma_r',   0.15)
        self.declare_parameter('eps_h',     0.10)
        self.declare_parameter('gate',      0.50)
        self.declare_parameter('max_range', 3.50)

        map_yaml       = self.get_parameter('map_yaml').value
        self.n_beams   = int(self.get_parameter('n_beams').value)
        self.sigma_qd  = float(self.get_parameter('sigma_qd').value)
        self.sigma_qth = float(self.get_parameter('sigma_qth').value)
        self.sigma_r   = float(self.get_parameter('sigma_r').value)
        self.eps_h     = float(self.get_parameter('eps_h').value)
        self.gate      = float(self.get_parameter('gate').value)
        self.max_range = float(self.get_parameter('max_range').value)

        if not map_yaml:
            raise RuntimeError('Parameter map_yaml muss gesetzt sein!')

        # Karte ──────────────────────────────────────────────────────────────
        self._load_map(Path(map_yaml))

        # EKF State ──────────────────────────────────────────────────────────
        self.x_ekf       = np.zeros(3)       # [x, y, yaw] in map
        self.P           = np.eye(3) * 1e6   # grosse Init-Unsicherheit bis Pose gesetzt
        self.initialized     = False
        self.prev_odom       = None
        self.beam_angles     = None
        self._last_odom_t    = None
        self._last_scan_wall = None   # wall-time des letzten /scan (fuer Watchdog)

        # TF ─────────────────────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # QoS fuer Karte (latched, damit spaete Subscriber sie noch erhalten)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Publishers ─────────────────────────────────────────────────────────
        self.pub_map       = self.create_publisher(OccupancyGrid, '/map',           map_qos)
        self.pub_pose      = self.create_publisher(PoseStamped,   '/ekf_pose',      10)
        self.pub_beams     = self.create_publisher(LaserScan,     '/ekf_beams',     10)
        self.pub_beams_hat = self.create_publisher(LaserScan,     '/ekf_beams_hat', 10)

        self._scan_count = 0

        # Subscribers ────────────────────────────────────────────────────────
        # qos_profile_sensor_data = BEST_EFFORT/VOLATILE → kompatibel mit TurtleBot3 bag
        self.create_subscription(
            Odometry, '/odom', self.cb_odom, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan', self.cb_scan, qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.cb_initialpose, 10)

        # TF-Heartbeat + Scan-Watchdog
        self.create_timer(0.05, self._heartbeat)

        self._publish_map()  # einmalig; TRANSIENT_LOCAL liefert sie an spaete Subscriber

        self.get_logger().info(
            'EKF Localizer bereit. '
            'Klicke in RViz auf "2D Pose Estimate" um die Startpose zu setzen.')

    # ── Karte ─────────────────────────────────────────────────────────────────

    def _load_map(self, yaml_path: Path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)

        self.map_res    = float(meta['resolution'])
        self.map_origin = [float(v) for v in meta['origin'][:2]]

        pgm_path = yaml_path.parent / meta['image']
        img = np.array(Image.open(pgm_path))
        self.map_h, self.map_w = img.shape

        occ_thresh  = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.25))
        self.occ_mask = img < int((1.0 - occ_thresh) * 255)

        self._map_img       = img
        self._occ_thresh    = occ_thresh
        self._free_thresh   = free_thresh

        self.get_logger().info(
            f'Karte: {yaml_path.name}  {self.map_w}x{self.map_h} px  '
            f'res={self.map_res} m/px')

    def _publish_map(self):
        img = self._map_img
        h, w = img.shape
        free_lim = 1.0 - self._free_thresh   # pixel/255 > das  → frei
        occ_lim  = 1.0 - self._occ_thresh    # pixel/255 < das  → belegt

        data = []
        # OccupancyGrid: Zeile 0 = unten (kleinstes y)
        # PGM: Zeile 0 = oben (groesstes y)  → umkehren
        for pgm_row in range(h - 1, -1, -1):
            for col in range(w):
                v = img[pgm_row, col] / 255.0
                if v > free_lim:
                    data.append(0)
                elif v < occ_lim:
                    data.append(100)
                else:
                    data.append(-1)

        msg = OccupancyGrid()
        msg.header.stamp              = self.get_clock().now().to_msg()
        msg.header.frame_id           = 'map'
        msg.info.resolution           = self.map_res
        msg.info.width                = w
        msg.info.height               = h
        msg.info.origin.position.x    = self.map_origin[0]
        msg.info.origin.position.y    = self.map_origin[1]
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        self.pub_map.publish(msg)
        self.get_logger().info('Karte auf /map gepublisht.')

    # ── Raycast / Messmodell ──────────────────────────────────────────────────

    def _world_to_arr(self, x, y):
        col = int((x - self.map_origin[0]) / self.map_res)
        row = self.map_h - 1 - int((y - self.map_origin[1]) / self.map_res)
        return col, row

    def _raycast(self, x, y, ang):
        cx, cy = np.cos(ang), np.sin(ang)
        step   = self.map_res * 0.5
        n      = int(self.max_range / step)
        for k in range(1, n + 1):
            d        = k * step
            col, row = self._world_to_arr(x + cx * d, y + cy * d)
            if col < 0 or col >= self.map_w or row < 0 or row >= self.map_h:
                return d
            if self.occ_mask[row, col]:
                return d
        return self.max_range

    def _expected_scan(self, pose):
        x, y, th = pose
        return np.array([self._raycast(x, y, th + a) for a in self.beam_angles])

    def _compute_H(self, pose):
        eps = self.eps_h
        H   = np.zeros((self.n_beams, 3))
        for j, dp in enumerate([
            np.array([eps, 0.0, 0.0]),
            np.array([0.0, eps, 0.0]),
            np.array([0.0, 0.0, eps]),
        ]):
            H[:, j] = (self._expected_scan(pose + dp)
                       - self._expected_scan(pose - dp)) / (2.0 * eps)
        return H

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_odom(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        cur  = np.array([p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w)])

        if self.prev_odom is None:
            self.prev_odom = cur
            return

        if not self.initialized:
            self.prev_odom = cur
            return

        # EKF Predict ────────────────────────────────────────────────────────
        dx   = cur[0] - self.prev_odom[0]
        dy   = cur[1] - self.prev_odom[1]
        dth  = float(np.arctan2(np.sin(cur[2] - self.prev_odom[2]),
                                np.cos(cur[2] - self.prev_odom[2])))
        dist = dx * np.cos(self.prev_odom[2]) + dy * np.sin(self.prev_odom[2])

        # Bag-Loop-Schutz: unrealistisch grosser Odom-Sprung → nur prev aktualisieren
        if abs(dist) > 0.3 or abs(dth) > 1.0:
            self.prev_odom = cur
            return

        th_new = float(np.arctan2(np.sin(self.x_ekf[2] + dth),
                                  np.cos(self.x_ekf[2] + dth)))
        self.x_ekf[0] += dist * np.cos(th_new)
        self.x_ekf[1] += dist * np.sin(th_new)
        self.x_ekf[2]  = th_new

        F  = np.array([[1.0, 0.0, -dist * np.sin(th_new)],
                       [0.0, 1.0,  dist * np.cos(th_new)],
                       [0.0, 0.0,  1.0]])
        G  = np.array([[np.cos(th_new), -dist * np.sin(th_new)],
                       [np.sin(th_new),  dist * np.cos(th_new)],
                       [0.0,             1.0]])
        Qu = np.diag([self.sigma_qd**2  * abs(dist),
                      self.sigma_qth**2 * abs(dth)])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T

        self.prev_odom = cur

    def cb_scan(self, msg: LaserScan):
        if not self.initialized:
            return

        # Beam-Indizes beim ersten Scan bestimmen
        if self.beam_angles is None:
            n   = len(msg.ranges)
            idx = np.linspace(0, n, self.n_beams, endpoint=False).astype(int)
            self.beam_angles = msg.angle_min + idx * msg.angle_increment

        idx   = np.linspace(0, len(msg.ranges), self.n_beams,
                            endpoint=False).astype(int)
        z_obs = np.array(msg.ranges, dtype=float)[idx]
        z_obs[~np.isfinite(z_obs)]    = self.max_range
        z_obs[z_obs < msg.range_min]  = self.max_range
        z_obs[z_obs > self.max_range] = self.max_range

        self._last_scan_wall = time.time()
        self._publish_beams(msg, z_obs, None)

        t0 = time.time()

        # EKF Update ─────────────────────────────────────────────────────────
        # Adaptive gate und clamp: bei grosser Unsicherheit (P gross) grosszuegiger,
        # bei kleiner Unsicherheit (P klein) konservativer.
        pos_std   = float(np.sqrt(np.trace(self.P[:2, :2])))  # aktuelle Positionsunsicherheit
        gate_eff  = self.gate + 2.0 * pos_std                 # gate waechst mit Unsicherheit
        max_corr  = float(np.clip(2.0 * pos_std, 0.15, 1.0))  # max Korrektur pro Scan

        z_hat = self._expected_scan(self.x_ekf)
        innov = z_obs - z_hat
        valid = ((z_obs  < self.max_range - 1e-3) &
                 (z_hat  < self.max_range - 1e-3) &
                 (np.abs(innov) < gate_eff))

        if valid.sum() >= 3:
            H_v = self._compute_H(self.x_ekf)[valid]
            i_v = innov[valid]
            R_v = np.eye(valid.sum()) * self.sigma_r ** 2
            S   = H_v @ self.P @ H_v.T + R_v
            try:
                K = self.P @ H_v.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                pass
            else:
                corr     = K @ i_v
                corr[:2] = np.clip(corr[:2], -max_corr, max_corr)
                corr[2]  = np.clip(corr[2],  -0.5,      0.5)
                self.x_ekf    = self.x_ekf + corr
                self.x_ekf[2] = float(np.arctan2(np.sin(self.x_ekf[2]),
                                                  np.cos(self.x_ekf[2])))
                self.P = (np.eye(3) - K @ H_v) @ self.P

        dt = (time.time() - t0) * 1000
        self._scan_count += 1

        # Timestamp-Delta: wie weit liegt der Scan-Stamp hinter der aktuellen Sim-Zeit?
        scan_t  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_t   = self.get_clock().now().nanoseconds * 1e-9
        ts_lag  = now_t - scan_t   # >0: Scan ist aelter als aktuelle Zeit (Queue-Lag)

        if self._scan_count % 5 == 0:
            self.get_logger().info(
                f'#{self._scan_count:4d} | '
                f'beams={valid.sum():2d}/{self.n_beams} | '
                f'gate={gate_eff:.2f}m | '
                f'dt={dt:.0f}ms | '
                f'lag={ts_lag*1000:+.0f}ms | '
                f'pos_std={pos_std:.3f}m | '
                f'x={self.x_ekf[0]:.2f} y={self.x_ekf[1]:.2f} '
                f'yaw={np.degrees(self.x_ekf[2]):.1f}deg'
            )

        if valid.sum() < 2:
            self.get_logger().warn(
                f'#{self._scan_count} KEIN UPDATE: nur {valid.sum()}/{self.n_beams} '
                f'Beams gueltig (gate={gate_eff:.2f}m, pos_std={pos_std:.3f}m, '
                f'lag={ts_lag*1000:+.0f}ms)')

        if abs(ts_lag) > 0.5:
            self.get_logger().warn(
                f'#{self._scan_count} TIMESTAMP-LAG: {ts_lag*1000:+.0f}ms '
                f'(Scan veraltet → TF-Lookup koennte fehlschlagen)')

        self._publish_beams(msg, z_obs, z_hat)
        self._publish_tf(msg.header.stamp)
        self._publish_pose(msg.header.stamp)

    def cb_initialpose(self, msg: PoseWithCovarianceStamped):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw  = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.x_ekf       = np.array([p.x, p.y, yaw])
        self.P           = np.diag([0.05**2, 0.05**2, 0.03**2])  # engere Init-Unsicherheit
        self.initialized = True
        self.get_logger().info(
            f'Startpose gesetzt:  x={p.x:.3f}  y={p.y:.3f}  '
            f'yaw={np.degrees(yaw):.1f} deg')

    # ── Beam-Visualisierung ───────────────────────────────────────────────────

    def _publish_beams(self, orig: LaserScan, z_obs: np.ndarray, z_hat):
        """Publisht die benutzten Strahlen als sparse LaserScan fuer RViz."""
        base = LaserScan()
        base.header      = orig.header          # frame_id = base_scan, selber Timestamp
        base.angle_min   = float(self.beam_angles[0])
        base.angle_max   = float(self.beam_angles[-1])
        base.angle_increment = float(
            (self.beam_angles[-1] - self.beam_angles[0]) / max(self.n_beams - 1, 1))
        base.range_min   = orig.range_min
        base.range_max   = orig.range_max

        base.ranges = z_obs.tolist()
        self.pub_beams.publish(base)

        if z_hat is not None:
            hat = LaserScan()
            hat.header           = base.header
            hat.angle_min        = base.angle_min
            hat.angle_max        = base.angle_max
            hat.angle_increment  = base.angle_increment
            hat.range_min        = base.range_min
            hat.range_max        = base.range_max
            hat.ranges = z_hat.tolist()
            self.pub_beams_hat.publish(hat)

    # ── TF + Pose publishen ───────────────────────────────────────────────────

    def _heartbeat(self):
        self._publish_tf()
        if self._last_scan_wall is None:
            return
        age = time.time() - self._last_scan_wall
        if age > 1.0:
            now_sim = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().warn(
                f'WATCHDOG: kein /scan seit {age:.1f}s wall-time | '
                f'sim_time={now_sim:.3f} | '
                f'sim_time laueft: {"ja" if age < 5.0 else "NEIN – bag beendet?"}')

    def _publish_tf(self, stamp=None):
        # stamp=None → Heartbeat-Timer (get_clock); stamp gesetzt → synchron zum Scan
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        if not self.initialized or self.prev_odom is None:
            t = TransformStamped()
            t.header.stamp    = stamp
            t.header.frame_id = 'map'
            t.child_frame_id  = 'odom'
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)
            return

        ix, iy, ith = inv_pose(*self.prev_odom)
        mx, my, mth = compose(self.x_ekf[0], self.x_ekf[1], self.x_ekf[2],
                              ix, iy, ith)
        qx, qy, qz, qw = yaw_to_quat(mth)

        t = TransformStamped()
        t.header.stamp        = stamp
        t.header.frame_id     = 'map'
        t.child_frame_id      = 'odom'
        t.transform.translation.x = float(mx)
        t.transform.translation.y = float(my)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _publish_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.x_ekf[0])
        msg.pose.position.y = float(self.x_ekf[1])
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(self.x_ekf[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub_pose.publish(msg)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = EKFLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
