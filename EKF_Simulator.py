"""
SCHRITT 1 -- Realdata Player (ohne EKF)
========================================
Spielt rosbag2_2 ab + zeigt:
  - Karte 2 als Hintergrund
  - GT-Pose (gruen, Cartographer map->base)
  - rohe Odometrie (rot, am ersten map->odom Frame ausgerichtet)
  - LiDAR-Vergleich: echter Scan vs. erwarteter Scan (Pixel-Raycast in der Karte)

Master-Takt = /scan (~5 Hz). Pro Scan wird die zeitlich naechste
/odom-Pose und der naechste map->odom Transform genommen.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from PIL import Image
import threading
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ─────────────────────────────── KONFIGURATION ────────────────────
BASE = Path(__file__).parent / "rosbag_day2"
BAG_PATH = BASE / "rosbag2_3"
MAP_YAML = BASE / "map3.yaml"
N_BEAMS = 15  # subsampled aus dem echten Scan
MAX_RANGE = 3.5  # m  (TurtleBot3 LDS)
ROBOT_R = 0.17

# EKF-Parameter (Defaults; ueber Slider tunbar)
SIGMA_QD = 0.08  # Prozessrauschen Distanz [m]
SIGMA_QTH = 0.05  # Prozessrauschen Yaw [rad]
SIGMA_R = 0.15  # erwartetes Mess-Rauschen pro Beam [m]
EPS_H = 0.10  # Schrittweite numerischer Jacobian (>= 2 Map-Pixel!) [m / rad]
GATE = 0.5  # Outlier-Gate fuer Innovation pro Beam [m]

typestore = get_typestore(Stores.ROS2_HUMBLE)


def quat_to_yaw(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny, cosy)


def compose(T_a_b, T_b_c):
    xa, ya, ta = T_a_b
    xb, yb, tb = T_b_c
    c, s = np.cos(ta), np.sin(ta)
    return (xa + c * xb - s * yb, ya + s * xb + c * yb, ta + tb)


# ─────────────────────────────── BAG LADEN ────────────────────────
print(f"Lese Bag: {BAG_PATH.name}")
odom_list, mo_list, scan_list = [], [], []
scan_meta = None
with Reader(BAG_PATH) as reader:
    for conn, ts, raw in reader.messages():
        t_msg = ts * 1e-9
        if conn.topic == "/odom":
            m = typestore.deserialize_cdr(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            odom_list.append((t_msg, p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w)))
        elif conn.topic == "/tf":
            m = typestore.deserialize_cdr(raw, conn.msgtype)
            for tf in m.transforms:
                if tf.header.frame_id == "map" and tf.child_frame_id == "odom":
                    t = tf.transform.translation
                    r = tf.transform.rotation
                    stp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                    mo_list.append((stp, t.x, t.y, quat_to_yaw(r.x, r.y, r.z, r.w)))
        elif conn.topic == "/scan":
            m = typestore.deserialize_cdr(raw, conn.msgtype)
            ranges = np.asarray(m.ranges, dtype=float)
            if scan_meta is None:
                scan_meta = (
                    m.angle_min,
                    m.angle_max,
                    m.angle_increment,
                    m.range_min,
                    m.range_max,
                )
            scan_list.append((t_msg, ranges))

odom_arr = np.array(odom_list)  # (N_o, 4)  [t,x,y,yaw]
mo_arr = np.array(mo_list)  # (N_m, 4)
scan_t = np.array([s[0] for s in scan_list])
scan_r = np.array([s[1] for s in scan_list])  # (N_s, 360)
print(f"  /odom: {len(odom_arr)}   map->odom: {len(mo_arr)}   /scan: {len(scan_r)}")
print(
    f"  scan: angle_min={scan_meta[0]:.3f} angle_inc={scan_meta[2]:.4f} "
    f"#beams={scan_r.shape[1]}"
)

# Beam-Subsampling: 12 gleichmaessig verteilte aus den 360
all_angles = scan_meta[0] + np.arange(scan_r.shape[1]) * scan_meta[2]
beam_idx = np.linspace(0, scan_r.shape[1], N_BEAMS, endpoint=False).astype(int)
BEAM_ANGLES = all_angles[beam_idx]  # (12,) Winkel im base_link Frame


# Master-Takt = /scan. Fuer jeden Scan: nearest odom + nearest map->odom
def nearest_idx(arr_t, t):
    return int(np.argmin(np.abs(arr_t - t)))


odom_t = odom_arr[:, 0]
mo_t = mo_arr[:, 0]

gt_poses = np.zeros((len(scan_r), 3))  # (x,y,yaw) in map
odom_poses = np.zeros(
    (len(scan_r), 3)
)  # rohe Odom in map (mit erstem map->odom angekleppt)
T_mo0 = (mo_arr[0, 1], mo_arr[0, 2], mo_arr[0, 3])
for i, t in enumerate(scan_t):
    io = nearest_idx(odom_t, t)
    im = nearest_idx(mo_t, t)
    odom_pose = (odom_arr[io, 1], odom_arr[io, 2], odom_arr[io, 3])
    T_mo_t = (mo_arr[im, 1], mo_arr[im, 2], mo_arr[im, 3])
    gt_poses[i] = compose(T_mo_t, odom_pose)
    odom_poses[i] = compose(T_mo0, odom_pose)

N_STEPS = len(scan_r)
DT_AVG = float(np.mean(np.diff(scan_t)))
DURATION = float(scan_t[-1] - scan_t[0])
print(f"  N_STEPS={N_STEPS}  DT_avg={DT_AVG * 1000:.0f} ms  Dauer={DURATION:.1f} s")

# Subsampled real scans (mit max-range / inf cleanup)
z_real = scan_r[:, beam_idx].copy()
z_real[~np.isfinite(z_real)] = MAX_RANGE
z_real[z_real < scan_meta[3]] = MAX_RANGE
z_real[z_real > MAX_RANGE] = MAX_RANGE


# ─────────────────────────────── KARTE LADEN ──────────────────────
with open(MAP_YAML) as f:
    meta = yaml.safe_load(f)
MAP_RES = meta["resolution"]
MAP_ORIGIN = meta["origin"][:2]
_pgm = MAP_YAML.parent / meta["image"]
if not _pgm.exists():
    _pgm = MAP_YAML.with_suffix(".pgm")
MAP_IMG = np.array(Image.open(_pgm))
MAP_H, MAP_W = MAP_IMG.shape
OCC_THRESH = meta.get("occupied_thresh", 0.65)
# In PGM: dunkle Pixel = belegt. Konvention: pixel/255 < (1 - occ_thresh) => occupied
OCC_MASK = MAP_IMG < int((1.0 - OCC_THRESH) * 255)
print(f"  Karte: {MAP_W}x{MAP_H} px, res={MAP_RES} m/px, occupied={OCC_MASK.sum()} px")


def world_to_px(x, y):
    """Welt(m) -> Plot-Koordinate passend zu imshow(origin='upper', extent=[0,W,0,H])."""
    return (x - MAP_ORIGIN[0]) / MAP_RES, (y - MAP_ORIGIN[1]) / MAP_RES


def world_to_arr(x, y):
    """Welt(m) -> Array-Index (col, row) im OCC_MASK (row=0 oben)."""
    col = int((x - MAP_ORIGIN[0]) / MAP_RES)
    row = MAP_H - 1 - int((y - MAP_ORIGIN[1]) / MAP_RES)
    return col, row


def raycast_pixel(x, y, ang, max_range=MAX_RANGE, step_m=MAP_RES * 0.25):
    """Pixel-basierter Raycast in OCC_MASK. Liefert Distanz in Meter."""
    cx, cy = np.cos(ang), np.sin(ang)
    n_steps = int(max_range / step_m)
    for k in range(1, n_steps + 1):
        d = k * step_m
        col, row = world_to_arr(x + cx * d, y + cy * d)
        if col < 0 or col >= MAP_W or row < 0 or row >= MAP_H:
            return d
        if OCC_MASK[row, col]:
            return d
    return max_range


def expected_scan(pose):
    x, y, th = pose
    return np.array([raycast_pixel(x, y, th + a) for a in BEAM_ANGLES])


# ─────────────────────────────── EKF (einmal vorab) ───────────────
def compute_H_num(pose, eps):
    H = np.zeros((N_BEAMS, 3))
    for col, dp in enumerate(
        [
            np.array([eps, 0.0, 0.0]),
            np.array([0.0, eps, 0.0]),
            np.array([0.0, 0.0, eps]),
        ]
    ):
        H[:, col] = (expected_scan(pose + dp) - expected_scan(pose - dp)) / (2 * eps)
    return H


def run_ekf(
    sigma_qd=SIGMA_QD,
    sigma_qth=SIGMA_QTH,
    sigma_r=SIGMA_R,
    eps_h=EPS_H,
    gate=GATE,
    verbose=True,
):
    if verbose:
        print(
            f"EKF: σ_qd={sigma_qd:.3f} σ_qth={sigma_qth:.3f} "
            f"σ_r={sigma_r:.3f} ε={eps_h:.3f} gate={gate:.2f}"
        )
    xs_e = np.zeros(N_STEPS)
    ys_e = np.zeros(N_STEPS)
    ths_e = np.zeros(N_STEPS)
    Ps = np.zeros((N_STEPS, 3, 3))

    x_ekf = gt_poses[0].astype(float).copy()
    P = np.diag([0.05**2, 0.05**2, 0.05**2])
    Q = np.diag([sigma_qd**2, sigma_qd**2, sigma_qth**2])
    I3 = np.eye(3)

    xs_e[0], ys_e[0], ths_e[0] = x_ekf
    Ps[0] = P

    odom_raw = np.zeros((N_STEPS, 3))
    for i, t in enumerate(scan_t):
        io = nearest_idx(odom_t, t)
        odom_raw[i] = odom_arr[io, 1:4]

    for i in range(1, N_STEPS):
        # Predict
        dx_o = odom_raw[i, 0] - odom_raw[i - 1, 0]
        dy_o = odom_raw[i, 1] - odom_raw[i - 1, 1]
        prev_yaw_odom = odom_raw[i - 1, 2]
        dist = dx_o * np.cos(prev_yaw_odom) + dy_o * np.sin(prev_yaw_odom)
        dth = (odom_raw[i, 2] - odom_raw[i - 1, 2] + np.pi) % (2 * np.pi) - np.pi

        th_old = x_ekf[2]
        th_new = (th_old + dth + np.pi) % (2 * np.pi) - np.pi

        # Update state: x = x + d*cos(th + dth), etc.
        x_ekf[0] += dist * np.cos(th_new)
        x_ekf[1] += dist * np.sin(th_new)
        x_ekf[2] = th_new

        # Jacobians
        F = np.array(
            [
                [1.0, 0.0, -dist * np.sin(th_new)],
                [0.0, 1.0, dist * np.cos(th_new)],
                [0.0, 0.0, 1.0],
            ]
        )
        G = np.array(
            [
                [np.cos(th_new), -dist * np.sin(th_new)],
                [np.sin(th_new), dist * np.cos(th_new)],
                [0.0, 1.0],
            ]
        )

        # Motion-dependent noise: uncertainty grows only when moving
        q_d_dynamic = sigma_qd**2 * np.abs(dist)
        q_th_dynamic = sigma_qth**2 * np.abs(dth)
        Qu = np.diag([q_d_dynamic, q_th_dynamic])
        P = F @ P @ F.T + G @ Qu @ G.T

        # Update
        z_obs = z_real[i]
        z_hat = expected_scan(x_ekf)
        innov = z_obs - z_hat
        valid = (
            (z_obs < MAX_RANGE - 1e-3)
            & (z_hat < MAX_RANGE - 1e-3)
            & (np.abs(innov) < gate)
        )

        if valid.sum() >= 3:
            H_full = compute_H_num(x_ekf, eps_h)
            H = H_full[valid]
            innov_v = innov[valid]
            R_v = np.eye(valid.sum()) * sigma_r**2
            S = H @ P @ H.T + R_v
            try:
                K = P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                xs_e[i], ys_e[i], ths_e[i] = x_ekf
                Ps[i] = P
                continue
            correction = K @ innov_v
            # Korrektur clampen, damit ein einzelner Sprung nicht alles zerstoert
            correction[:2] = np.clip(correction[:2], -0.3, 0.3)
            correction[2] = np.clip(correction[2], -0.3, 0.3)
            x_ekf = x_ekf + correction
            x_ekf[2] = (x_ekf[2] + np.pi) % (2 * np.pi) - np.pi
            P = (I3 - K @ H) @ P

        xs_e[i], ys_e[i], ths_e[i] = x_ekf
        Ps[i] = P
        _ekf_progress[0] = i  # fuer Progress-Bar im Hauptthread

    if verbose:
        err = np.hypot(xs_e - gt_poses[:, 0], ys_e - gt_poses[:, 1])
        err_o = np.hypot(
            odom_poses[:, 0] - gt_poses[:, 0], odom_poses[:, 1] - gt_poses[:, 1]
        )
        print(
            f"  RMSE  EKF={np.sqrt(np.mean(err**2)):.3f} m   "
            f"Odom={np.sqrt(np.mean(err_o**2)):.3f} m"
        )
    return xs_e, ys_e, ths_e, Ps


# Ergebnis-Container fuer Thread-Kommunikation
_ekf_result = [None]  # (xs_e, ys_e, ths_e, Ps) wenn fertig
_ekf_progress = [0]  # aktueller Step (0 .. N_STEPS)
_ekf_running = [False]


def _run_ekf_thread(kw):
    _ekf_running[0] = True
    _ekf_progress[0] = 0
    # run_ekf mit Progress-Callback
    result = run_ekf(**kw)
    _ekf_result[0] = result
    _ekf_running[0] = False


# ─────────────────────────────── FIGURE ───────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
ax = fig.add_axes([0.03, 0.30, 0.44, 0.65])
ax.set_facecolor("#1a1a2e")

ax.imshow(
    MAP_IMG,
    cmap="gray",
    origin="upper",
    extent=[0, MAP_W, 0, MAP_H],
    vmin=0,
    vmax=255,
    zorder=0,
)
ax.set_xlim(0, MAP_W)
ax.set_ylim(0, MAP_H)
ax.set_aspect("equal")
ax.tick_params(colors="white", labelsize=7)
for sp in ax.spines.values():
    sp.set_edgecolor("#444466")
ax.set_title(
    f"{BAG_PATH.name} auf {MAP_YAML.stem}   GT (gruen)  Odom (rot)",
    color="white",
    fontsize=11,
)

# Vollstaendige Pfade (transparent, als Referenz)
gx, gy = world_to_px(gt_poses[:, 0], gt_poses[:, 1])
ox, oy = world_to_px(odom_poses[:, 0], odom_poses[:, 1])
ax.plot(gx, gy, "-", color="#00ff88", lw=0.9, alpha=0.25, zorder=2)
ax.plot(ox, oy, "--", color="#ff4444", lw=0.9, alpha=0.25, zorder=2)
# EKF-Pfad (erst leer, wird nach Berechnung befuellt)
xs_e = np.zeros(N_STEPS)
ys_e = np.zeros(N_STEPS)
ths_e = np.zeros(N_STEPS)
Ps = np.zeros((N_STEPS, 3, 3))
ex_full, ey_full = world_to_px(xs_e, ys_e)
(ekf_full_line,) = ax.plot(
    ex_full, ey_full, "-", color="#44aaff", lw=0.9, alpha=0.25, zorder=2
)

# animierte Trails
(trail_gt,) = ax.plot(
    [], [], "-", color="#00ff88", lw=1.8, alpha=0.95, zorder=4, label="Ground Truth"
)
(trail_odom,) = ax.plot(
    [], [], "--", color="#ff4444", lw=1.5, alpha=0.85, zorder=4, label="Odometrie"
)
(trail_ekf,) = ax.plot(
    [], [], "-", color="#44aaff", lw=2.0, alpha=0.95, zorder=4, label="EKF"
)

# Pixelradius des Roboters
robot_px = ROBOT_R / MAP_RES
body_gt = patches.Circle((0, 0), robot_px, color="#00ff88", alpha=0.20, zorder=5)
body_odom = patches.Circle((0, 0), robot_px, color="#ff4444", alpha=0.18, zorder=5)
body_ekf = patches.Circle((0, 0), robot_px, color="#44aaff", alpha=0.22, zorder=5)
ax.add_patch(body_gt)
ax.add_patch(body_odom)
ax.add_patch(body_ekf)

(dot_gt,) = ax.plot([], [], "o", color="#00ff88", ms=8, zorder=6)
(dot_odom,) = ax.plot([], [], "o", color="#ff4444", ms=8, zorder=6)
(dot_ekf,) = ax.plot([], [], "o", color="#44aaff", ms=9, zorder=6)
dir_gt = ax.quiver(
    [], [], [], [], color="#00ff88", scale=15, width=0.006, headwidth=4, zorder=7
)
dir_odom = ax.quiver(
    [], [], [], [], color="#ff4444", scale=15, width=0.006, headwidth=4, zorder=7
)
dir_ekf = ax.quiver(
    [], [], [], [], color="#44aaff", scale=15, width=0.007, headwidth=4, zorder=7
)

# Kovarianz-Ellipse fuer EKF
(ellipse_line,) = ax.plot([], [], "-", color="#44aaff", lw=1.2, alpha=0.7, zorder=5)

ax.legend(
    loc="lower right",
    facecolor="#0f3460",
    labelcolor="white",
    fontsize=8,
    framealpha=0.85,
)

# echter Scan auf der Karte
(real_scan_dots,) = ax.plot([], [], ".", color="#ffee44", ms=4, alpha=0.85, zorder=8)

info_txt = ax.text(
    0.02,
    0.97,
    "",
    transform=ax.transAxes,
    fontsize=9,
    color="white",
    va="top",
    bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.8),
)


# ─────────────────────────────── LIDAR-PANEL (rechts oben) ────────
ax_lidar = fig.add_axes([0.51, 0.30, 0.46, 0.65])
ax_lidar.set_facecolor("#1a1a2e")
ax_lidar.imshow(
    MAP_IMG,
    cmap="gray",
    origin="upper",
    extent=[0, MAP_W, 0, MAP_H],
    vmin=0,
    vmax=255,
    zorder=0,
)
ax_lidar.set_xlim(0, MAP_W)
ax_lidar.set_ylim(0, MAP_H)
ax_lidar.set_aspect("equal")
ax_lidar.tick_params(colors="white", labelsize=7)
for sp in ax_lidar.spines.values():
    sp.set_edgecolor("#444466")
ax_lidar.set_title(
    "LiDAR-Vergleich  ·  echt von GT (gruen)  vs.  erwartet von Odom (rot)",
    color="white",
    fontsize=10,
)

# Linien-Container: pro Beam eine Linie, vorab erstellt
real_lines, hat_lines = [], []
for _ in range(N_BEAMS):
    (l1,) = ax_lidar.plot([], [], "-", color="#00ff88", lw=1.0, alpha=0.7, zorder=3)
    (l2,) = ax_lidar.plot([], [], "--", color="#ff4444", lw=1.0, alpha=0.7, zorder=3)
    real_lines.append(l1)
    hat_lines.append(l2)
(gt_dot,) = ax_lidar.plot(
    [],
    [],
    "o",
    color="#00ff88",
    ms=8,
    zorder=6,
    markeredgecolor="white",
    markeredgewidth=0.8,
)
(odom_dot,) = ax_lidar.plot(
    [],
    [],
    "o",
    color="#ff4444",
    ms=8,
    zorder=6,
    markeredgecolor="white",
    markeredgewidth=0.8,
)
lidar_info = ax_lidar.text(
    0.02,
    0.97,
    "",
    transform=ax_lidar.transAxes,
    fontsize=9,
    color="white",
    va="top",
    bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.8),
)


# ─────────────────────────────── SLIDER + PLAY ────────────────────
ax_t = fig.add_axes([0.06, 0.22, 0.88, 0.022], facecolor="#0f3460")
sl_time = Slider(ax_t, "Zeit (s)", 0.0, DURATION, valinit=0.0, color="#00ff88")
sl_time.label.set_color("white")
sl_time.valtext.set_color("white")


def make_slider(rect, label, vmin, vmax, vinit, color):
    a = fig.add_axes(rect, facecolor="#0f3460")
    sl = Slider(a, label, vmin, vmax, valinit=vinit, color=color)
    sl.label.set_color("white")
    sl.valtext.set_color("white")
    return sl


# EKF-Tuning-Slider (links unten)
sl_qd = make_slider(
    [0.08, 0.16, 0.36, 0.020], "σ_Q dist", 0.005, 0.40, SIGMA_QD, "#44aaff"
)
sl_qth = make_slider(
    [0.08, 0.13, 0.36, 0.020], "σ_Q yaw", 0.005, 0.30, SIGMA_QTH, "#44aaff"
)
sl_r = make_slider(
    [0.08, 0.10, 0.36, 0.020], "σ_R lidar", 0.02, 0.50, SIGMA_R, "#ffee44"
)
sl_eps = make_slider(
    [0.08, 0.07, 0.36, 0.020], "ε Jacobian", 0.05, 0.30, EPS_H, "#ff8844"
)
sl_gate = make_slider(
    [0.08, 0.04, 0.36, 0.020], "gate (m)", 0.10, 2.00, GATE, "#aa44ff"
)

ax_play = fig.add_axes([0.55, 0.04, 0.10, 0.035])
btn_play = Button(ax_play, "Play / Pause", color="#1a3a1a", hovercolor="#2a6a2a")
btn_play.label.set_color("#00ff88")
btn_play.label.set_fontsize(9)

ax_recompute = fig.add_axes([0.67, 0.04, 0.14, 0.035])
btn_recompute = Button(
    ax_recompute, "EKF neu rechnen", color="#3a1a1a", hovercolor="#6a2a2a"
)
btn_recompute.label.set_color("#ffaa44")
btn_recompute.label.set_fontsize(9)

ax_stats = fig.add_axes([0.83, 0.04, 0.14, 0.16])
ax_stats.set_facecolor("#0f3460")
ax_stats.axis("off")
stats_txt = ax_stats.text(
    0.05,
    0.95,
    "",
    transform=ax_stats.transAxes,
    fontsize=9,
    color="white",
    va="top",
    family="monospace",
)


# ─────────────────────────────── DRAW ─────────────────────────────
def step_to_idx(val):
    # Slider zeigt Sekunden seit Bag-Start, Mapping per linearer Suche
    return int(np.clip(np.searchsorted(scan_t - scan_t[0], val), 0, N_STEPS - 1))


def cov_ellipse_pts(cx, cy, cov2, n_std=2, n_pts=60):
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 1e-12)
    a = n_std * np.sqrt(vals[1])
    b = n_std * np.sqrt(vals[0])
    angle = np.arctan2(vecs[1, 1], vecs[0, 1])
    t = np.linspace(0, 2 * np.pi, n_pts)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    c, s = np.cos(angle), np.sin(angle)
    return ex * c - ey * s + cx, ex * s + ey * c + cy


def draw_frame(i):
    i = int(np.clip(i, 0, N_STEPS - 1))
    trail_gt.set_data(gx[: i + 1], gy[: i + 1])
    trail_odom.set_data(ox[: i + 1], oy[: i + 1])
    trail_ekf.set_data(ex_full[: i + 1], ey_full[: i + 1])

    gt_x, gt_y, gt_th = gt_poses[i]
    od_x, od_y, od_th = odom_poses[i]
    ek_x, ek_y, ek_th = xs_e[i], ys_e[i], ths_e[i]
    gpx, gpy = world_to_px(gt_x, gt_y)
    opx, opy = world_to_px(od_x, od_y)
    epx, epy = world_to_px(ek_x, ek_y)

    body_gt.center = (gpx, gpy)
    body_odom.center = (opx, opy)
    body_ekf.center = (epx, epy)
    dot_gt.set_data([gpx], [gpy])
    dot_odom.set_data([opx], [opy])
    dot_ekf.set_data([epx], [epy])
    dir_gt.set_offsets([[gpx, gpy]])
    dir_gt.set_UVC(np.cos(gt_th), np.sin(gt_th))
    dir_odom.set_offsets([[opx, opy]])
    dir_odom.set_UVC(np.cos(od_th), np.sin(od_th))
    dir_ekf.set_offsets([[epx, epy]])
    dir_ekf.set_UVC(np.cos(ek_th), np.sin(ek_th))

    # Kovarianz-Ellipse (in Welt-m, dann in Pixel)
    ex_w, ey_w = cov_ellipse_pts(ek_x, ek_y, Ps[i][0:2, 0:2])
    epx_e, epy_e = world_to_px(ex_w, ey_w)
    ellipse_line.set_data(epx_e, epy_e)

    # Echter Scan als Punktwolke (von GT-Pose)
    rs = z_real[i]
    valid = rs < MAX_RANGE - 1e-3
    rx = gt_x + rs[valid] * np.cos(gt_th + BEAM_ANGLES[valid])
    ry = gt_y + rs[valid] * np.sin(gt_th + BEAM_ANGLES[valid])
    rpx, rpy = world_to_px(rx, ry)
    real_scan_dots.set_data(rpx, rpy)

    err_o = np.hypot(gt_x - od_x, gt_y - od_y)
    err_e = np.hypot(gt_x - ek_x, gt_y - ek_y)
    sig_x = np.sqrt(Ps[i][0, 0])
    sig_y = np.sqrt(Ps[i][1, 1])
    info_txt.set_text(
        f"step {i}/{N_STEPS - 1}   t={scan_t[i] - scan_t[0]:.2f} s\n"
        f"GT   x={gt_x:+.2f} y={gt_y:+.2f} yaw={np.degrees(gt_th):+6.1f}°\n"
        f"odom x={od_x:+.2f} y={od_y:+.2f} yaw={np.degrees(od_th):+6.1f}°\n"
        f"EKF  x={ek_x:+.2f} y={ek_y:+.2f} yaw={np.degrees(ek_th):+6.1f}°\n"
        f"|GT-Odom|={err_o:.3f} m   |GT-EKF|={err_e:.3f} m\n"
        f"σ_x={sig_x:.3f}  σ_y={sig_y:.3f}"
    )

    # Rechtes Panel: echter Scan (von GT) vs. erwarteter Scan (von Odom)
    z_hat = expected_scan(odom_poses[i])
    gpx, gpy = world_to_px(gt_x, gt_y)
    opx2, opy2 = world_to_px(od_x, od_y)
    gt_dot.set_data([gpx], [gpy])
    odom_dot.set_data([opx2], [opy2])
    for k in range(N_BEAMS):
        # echter Strahl: vom GT-Ort, GT-Yaw
        a_gt = gt_th + BEAM_ANGLES[k]
        r = rs[k]
        ex, ey = world_to_px(gt_x + r * np.cos(a_gt), gt_y + r * np.sin(a_gt))
        real_lines[k].set_data([gpx, ex], [gpy, ey])
        # erwarteter Strahl: vom Odom-Ort, Odom-Yaw
        a_od = od_th + BEAM_ANGLES[k]
        r2 = z_hat[k]
        hx, hy = world_to_px(od_x + r2 * np.cos(a_od), od_y + r2 * np.sin(a_od))
        hat_lines[k].set_data([opx2, hx], [opy2, hy])

    diff = z_hat - rs
    lidar_info.set_text(
        f"step {i}   beams={N_BEAMS}\n"
        f"|GT-Odom| pose-Drift = {err_o:.3f} m\n"
        f"mean(z_hat - z_real) = {np.mean(diff):+.3f} m\n"
        f"max |diff|           = {np.max(np.abs(diff)):.3f} m"
    )

    fig.canvas.draw_idle()


# ─────────────────────────────── CALLBACKS ────────────────────────
playing = [False]


def on_time(val):
    if not playing[0]:
        draw_frame(step_to_idx(val))


sl_time.on_changed(on_time)


def on_play(event):
    playing[0] = not playing[0]
    if playing[0]:
        anim.event_source.start()
    else:
        anim.event_source.stop()


btn_play.on_clicked(on_play)


def update_stats():
    err_e = np.hypot(xs_e - gt_poses[:, 0], ys_e - gt_poses[:, 1])
    err_o = np.hypot(
        odom_poses[:, 0] - gt_poses[:, 0], odom_poses[:, 1] - gt_poses[:, 1]
    )
    stats_txt.set_text(
        f"RMSE  [m]\n"
        f"  Odom: {np.sqrt(np.mean(err_o**2)):.3f}\n"
        f"  EKF : {np.sqrt(np.mean(err_e**2)):.3f}\n\n"
        f"max err [m]\n"
        f"  Odom: {err_o.max():.3f}\n"
        f"  EKF : {err_e.max():.3f}"
    )


# ── Progress-Bar Overlay (unsichtbar bis EKF laeuft) ──────────────
ax_prog = fig.add_axes([0.25, 0.47, 0.50, 0.06])
ax_prog.set_facecolor("#0a0a1e")
ax_prog.set_xlim(0, 1)
ax_prog.set_ylim(0, 1)
ax_prog.axis("off")
ax_prog.set_visible(False)
for sp in ax_prog.spines.values():
    sp.set_edgecolor("#44aaff")
    sp.set_linewidth(2)

# Hintergrund-Balken (grau)
prog_bg = patches.FancyBboxPatch(
    (0.02, 0.25),
    0.96,
    0.50,
    boxstyle="round,pad=0.01",
    facecolor="#1a1a3e",
    edgecolor="#44aaff",
    linewidth=2,
    transform=ax_prog.transAxes,
    zorder=1,
)
ax_prog.add_patch(prog_bg)

# Fortschritts-Balken (blau, Breite wird animiert)
prog_bar = patches.FancyBboxPatch(
    (0.02, 0.25),
    0.0,
    0.50,
    boxstyle="round,pad=0.01",
    facecolor="#44aaff",
    edgecolor="none",
    transform=ax_prog.transAxes,
    zorder=2,
)
ax_prog.add_patch(prog_bar)

prog_txt = ax_prog.text(
    0.50,
    0.50,
    "EKF wird berechnet ...",
    transform=ax_prog.transAxes,
    ha="center",
    va="center",
    fontsize=11,
    color="white",
    fontweight="bold",
    zorder=3,
)


def _show_progress(visible):
    ax_prog.set_visible(visible)
    # Haupt-Panels und Slider abdunkeln waehrend Berechnung
    for a in (ax, ax_lidar):
        a.set_alpha(0.3 if visible else 1.0)
    fig.canvas.draw_idle()


def on_recompute(event):
    if _ekf_running[0]:
        return
    _ekf_result[0] = None
    _ekf_progress[0] = 0
    prog_bar.set_width(0.0)
    prog_txt.set_text("EKF wird berechnet ...  0 %")
    _show_progress(True)
    kw = dict(
        sigma_qd=sl_qd.val,
        sigma_qth=sl_qth.val,
        sigma_r=sl_r.val,
        eps_h=sl_eps.val,
        gate=sl_gate.val,
        verbose=True,
    )
    t = threading.Thread(target=_run_ekf_thread, args=(kw,), daemon=True)
    t.start()


btn_recompute.on_clicked(on_recompute)


def _poll_ekf(_frame):
    """Wird von FuncAnimation aufgerufen – prueft ob EKF-Thread fertig ist."""
    global xs_e, ys_e, ths_e, Ps, ex_full, ey_full
    if _ekf_running[0]:
        pct = _ekf_progress[0] / max(N_STEPS - 1, 1)
        prog_bar.set_width(0.96 * pct)
        prog_txt.set_text(f"EKF wird berechnet ...  {pct * 100:.0f} %")
        fig.canvas.draw_idle()
        return
    if _ekf_result[0] is not None:
        xs_e, ys_e, ths_e, Ps = _ekf_result[0]
        _ekf_result[0] = None
        ex_full, ey_full = world_to_px(xs_e, ys_e)
        ekf_full_line.set_data(ex_full, ey_full)
        update_stats()
        _show_progress(False)
        draw_frame(step_to_idx(sl_time.val))


def animate(_frame):
    _poll_ekf(_frame)
    if not playing[0]:
        return
    i = step_to_idx(sl_time.val) + 1
    if i >= N_STEPS:
        playing[0] = False
        anim.event_source.stop()
        return
    sl_time.eventson = False
    sl_time.set_val(scan_t[i] - scan_t[0])
    sl_time.eventson = True
    draw_frame(i)


anim = FuncAnimation(
    fig,
    animate,
    interval=int(DT_AVG * 1000),
    blit=False,
    repeat=False,
    cache_frame_data=False,
)

# Initialer EKF-Lauf ueber Thread
on_recompute(None)

plt.show()
