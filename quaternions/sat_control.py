#!/usr/bin/env python3
# (See full docstring in previous attempt; same functionality.)
import math, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SIMULATION_DT_S = 0.01

def q_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def q_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def q_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def q_from_euler_xyz(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return q_normalize(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ], dtype=float))

def q_rotate(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = vq
    a = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)
    qc = np.array([w1, -x1, -y1, -z1], dtype=float)
    w3, x3, y3, z3 = a
    w4, x4, y4, z4 = qc
    out = np.array([
        w3*w4 - x3*x4 - y3*y4 - z3*z4,
        w3*x4 + x3*w4 + y3*z4 - z3*y4,
        w3*y4 - x3*z4 + y3*w4 + z3*x4,
        w3*z4 + x3*y4 - y3*x4 + z3*w4
    ], dtype=float)
    return out[1:]

def omega_to_qdot(q, omega):
    w, x, y, z = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        -x*wx - y*wy - z*wz,
         w*wx + y*wz - z*wy,
         w*wy - x*wz + z*wx,
         w*wz + x*wy - y*wx
    ], dtype=float)

def integrate_q(q, omega, dt):
    def f(qq): return omega_to_qdot(qq, omega)
    k1 = f(q)
    k2 = f(q + 0.5*dt*k1)
    k3 = f(q + 0.5*dt*k2)
    k4 = f(q + dt*k3)
    q_new = q + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return q_normalize(q_new)

def q_err(q_des, q):
    w1, x1, y1, z1 = q_des
    w2, x2, y2, z2 = q
    qc = np.array([w2, -x2, -y2, -z2], dtype=float)
    w3, x3, y3, z3 = w1, x1, y1, z1
    w4, x4, y4, z4 = qc
    return np.array([
        w3*w4 - x3*x4 - y3*y4 - z3*z4,
        w3*x4 + x3*w4 + y3*z4 - z3*y4,
        w3*y4 - x3*z4 + y3*w4 + z3*x4,
        w3*z4 + x3*y4 - y3*x4 + z3*w4
    ], dtype=float)

try:
    from adcs_controller import ADCSController  # type: ignore
    _HAS_EXTERNAL_CONTROLLER = True
except Exception:
    _HAS_EXTERNAL_CONTROLLER = False

class _FallbackADCSController:
    
    def __init__(self, kp=2.0, kd=0.6, omega_cmd_max=30.0):
        self.kp = kp
        self.kd = kd
        self.omega_cmd_max = omega_cmd_max  # optional safety clamp

    def step(self, quat_ref, body_rate_rad_per_s, q_current, omega_des=np.zeros(3)):
        q_current = q_normalize(q_current)
        # Body-frame error: rotate current → ref, expressed in body frame
        qe = q_mul(q_conj(q_current), quat_ref)      # NOTE: order swapped!
        s = 1.0 if qe[0] >= 0 else -1.0
        e_vec = 2.0 * s * qe[1:]                     # small-angle axis-angle (body frame)

        # Proper D term uses relative rate (ω - ω_des) in body frame
        omega_err = np.asarray(body_rate_rad_per_s, float) - np.asarray(omega_des, float)

        omega_cmd = self.kp * e_vec - self.kd * omega_err

        # (Optional) clip to actuator limits for robustness
        n = np.linalg.norm(omega_cmd)
        if n > self.omega_cmd_max:
            omega_cmd = omega_cmd * (self.omega_cmd_max / n)
        return omega_cmd

controller = ADCSController() if _HAS_EXTERNAL_CONTROLLER else _FallbackADCSController()

class RigidBody:
    def __init__(self, J=np.diag([0.03, 0.04, 0.05])):
        self.J = J
        self.Jinv = np.linalg.inv(J)
        self.q = np.array([1.0,0.0,0.0,0.0])
        self.omega = np.zeros(3)
        self.rate_tau = 0.08
    def step(self, omega_cmd, dt):
        domega = (omega_cmd - self.omega) / self.rate_tau
        self.omega = self.omega + dt * domega
        self.q = integrate_q(self.q, self.omega, dt)

def cube_vertices(edge=0.6):
    e = edge/2.0
    V = np.array([
        [-e,-e,-e],[ e,-e,-e],[ e, e,-e],[-e, e,-e],
        [-e,-e, e],[ e,-e, e],[ e, e, e],[-e, e, e],
    ], dtype=float)
    faces = [
        [0,1,2,3],[4,5,6,7],[0,1,5,4],
        [2,3,7,6],[1,2,6,5],[0,3,7,4]
    ]
    return V, faces

def panel_vertices(width=0.5, height=0.25, offset=0.4):
    w = width/2.0; h = height/2.0
    Pp = np.array([[-w, offset, -h],[ w, offset, -h],[ w, offset,  h],[-w, offset,  h]], dtype=float)
    Pm = np.array([[-w,-offset, -h],[ w,-offset, -h],[ w,-offset,  h],[-w,-offset,  h]], dtype=float)
    return Pp, Pm

def rotate_points(q, pts):
    return np.array([q_rotate(q, p) for p in pts])

sat = RigidBody()
q_ref = np.array([1.0,0.0,0.0,0.0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.canvas.manager.set_window_title("Satellite Attitude Simulator (Quaternion)")

axis_len = 1.0
ax.quiver(0,0,0, axis_len,0,0); ax.text(axis_len,0,0,"Wx")
ax.quiver(0,0,0, 0,axis_len,0); ax.text(0,axis_len,0,"Wy")
ax.quiver(0,0,0, 0,0,axis_len); ax.text(0,0,axis_len,"Wz")

V, faces = cube_vertices(edge=0.6)
Pp, Pm = panel_vertices(width=0.5, height=0.25, offset=0.4)
cube_poly = Poly3DCollection([], alpha=0.4)
panel_poly_1 = Poly3DCollection([], alpha=0.6)
panel_poly_2 = Poly3DCollection([], alpha=0.6)
ax.add_collection3d(cube_poly)
ax.add_collection3d(panel_poly_1)
ax.add_collection3d(panel_poly_2)

bx = ax.quiver(0,0,0, 0,0,0)
by = ax.quiver(0,0,0, 0,0,0)
bz = ax.quiver(0,0,0, 0,0,0)

body_tip_scatter = ax.scatter([], [], [])
ref_tip_scatter = ax.scatter([], [], [])

ax.set_xlim([-1.2, 1.2]); ax.set_ylim([-1.2, 1.2]); ax.set_zlim([-1.2, 1.2])
ax.set_box_aspect([1,1,1])
ax.set_title("Cube + panels; body axes; tips of body x (●) & ref x (○)")

slider_ax_r = plt.axes([0.15, 0.02, 0.7, 0.02])
slider_ax_p = plt.axes([0.15, 0.05, 0.7, 0.02])
slider_ax_y = plt.axes([0.15, 0.08, 0.7, 0.02])
s_roll = Slider(slider_ax_r, "Roll(deg)", -180.0, 180.0, valinit=0.0)
s_pitch = Slider(slider_ax_p, "Pitch(deg)", -180.0, 180.0, valinit=0.0)
s_yaw = Slider(slider_ax_y, "Yaw(deg)", -180.0, 180.0, valinit=0.0)

def get_ref_quat():
    r = math.radians(s_roll.val); p = math.radians(s_pitch.val); y = math.radians(s_yaw.val)
    return q_from_euler_xyz(r, p, y)

def update_plot():
    Vb = rotate_points(sat.q, V)
    cube_faces = [[Vb[idx] for idx in face] for face in faces]
    cube_poly.set_verts(cube_faces)
    Pp_b = rotate_points(sat.q, Pp)
    Pm_b = rotate_points(sat.q, Pm)
    panel_poly_1.set_verts([Pp_b])
    panel_poly_2.set_verts([Pm_b])
    ex = q_rotate(sat.q, np.array([1.0,0,0]))
    ey = q_rotate(sat.q, np.array([0,1.0,0]))
    ez = q_rotate(sat.q, np.array([0,0,1.0]))
    global bx, by, bz
    try:
        bx.remove(); by.remove(); bz.remove()
    except Exception:
        pass
    bx = ax.quiver(0,0,0, ex[0],ex[1],ex[2])
    by = ax.quiver(0,0,0, ey[0],ey[1],ey[2])
    bz = ax.quiver(0,0,0, ez[0],ez[1],ez[2])
    body_tip = ex
    ref_q = get_ref_quat()
    ref_tip = q_rotate(ref_q, np.array([1.0,0,0]))
    body_tip_scatter._offsets3d = ([body_tip[0]],[body_tip[1]],[body_tip[2]])
    ref_tip_scatter._offsets3d = ([ref_tip[0]],[ref_tip[1]],[ref_tip[2]])

def sim_step(_frame):
    quat_ref = get_ref_quat()
    omega_cmd = controller.step(quat_ref, sat.omega, sat.q)
    sat.step(omega_cmd, SIMULATION_DT_S)
    update_plot()
    return []

update_plot()
ani = FuncAnimation(fig, sim_step, interval=int(SIMULATION_DT_S*1000), blit=False)
plt.show()