#!/usr/bin/env python3
"""
Quaternion Toys for Roboticists
--------------------------------

Run any of these exercises:

1) Composition demo (frame rotations order matters)
   python quaternion_toys.py compose --deg1 90 --axis1 0 0 1 --deg2 90 --axis2 1 0 0

2) SLERP path (visualize interpolated orientations)
   python quaternion_toys.py slerp --q0_euler_deg 0 0 0 --q1_euler_deg 0 90 0 --steps 25

3) Attitude servo (error quaternion -> omega command -> integrate)
   python quaternion_toys.py servo --q_des_euler_deg 0 45 0 --kp 3.0 --kd 0.6 --tfinal 3.0 --dt 0.01

By default the script uses the *reference solutions*. Edit the TODO blocks in the
"student_*" functions and run with --use-student to test your code.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, acos, sqrt, pi

# ----------------------------
# Quaternion utility routines
# ----------------------------
def q_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def q_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def q_mul(q1, q2):
    """Hamilton product: compose rotations (apply q2 then q1 if using v' = q*v*q*)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def q_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = sin(angle_rad/2.0)
    return q_normalize(np.array([cos(angle_rad/2.0), *(axis*s)]))

def q_rotate(q, v):
    """Rotate vector v (3,) by quaternion q as v' = q * (0,v) * q*."""
    vq = np.array([0.0, *v], dtype=float)
    return (q_mul(q_mul(q, vq), q_conj(q)))[1:]

def q_from_euler_xyz(roll, pitch, yaw):
    """Extrinsic XYZ (roll->pitch->yaw) to quaternion (common in robotics)."""
    cr, sr = cos(roll/2), sin(roll/2)
    cp, sp = cos(pitch/2), sin(pitch/2)
    cy, sy = cos(yaw/2), sin(yaw/2)
    # R = Rx(roll) * Ry(pitch) * Rz(yaw)
    return q_normalize(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ], dtype=float))

def euler_xyz_from_q(q):
    """Inverse of q_from_euler_xyz."""
    w, x, y, z = q
    # roll (x-axis rotation)
    t0 = +2.0*(w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)
    # pitch (y-axis rotation)
    t2 = +2.0*(w*y - z*x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    # yaw (z-axis rotation)
    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw])

def q_slerp(q0, q1, u):
    """Spherical linear interpolation between unit quaternions q0 and q1 at fraction u in [0,1]."""
    q0 = q_normalize(q0)
    q1 = q_normalize(q1)
    dot = np.dot(q0, q1)
    # Take the short path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    # If the quaternions are close, use linear interpolation
    if dot > 0.9995:
        return q_normalize(q0 + u*(q1 - q0))
    theta_0 = np.arccos(dot)
    theta = theta_0 * u
    q2 = q_normalize(q1 - q0*dot)
    return q_normalize(q0*np.cos(theta) + q2*np.sin(theta))

def omega_to_qdot(q, omega_body):
    """Quaternion derivative from body-frame angular rate omega (rad/s)."""
    w, x, y, z = q
    wx, wy, wz = omega_body
    # qdot = 0.5 * q ⊗ [0, ω]
    qdot = 0.5 * np.array([
        -x*wx - y*wy - z*wz,
         w*wx + y*wz - z*wy,
         w*wy - x*wz + z*wx,
         w*wz + x*wy - y*wx
    ])
    return qdot

def integrate_q(q, omega_body, dt):
    """RK4 integration of quaternion dynamics with constant omega over dt."""
    def f(qq): return omega_to_qdot(qq, omega_body)
    k1 = f(q)
    k2 = f(q + 0.5*dt*k1)
    k3 = f(q + 0.5*dt*k2)
    k4 = f(q + dt*k3)
    q_new = q + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return q_normalize(q_new)

def q_err(q_des, q):
    """Error quaternion that takes q to q_des: q_err ⊗ q = q_des."""
    return q_mul(q_des, q_conj(q))

def axis_angle_from_q(q):
    q = q_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2*np.arccos(w)
    s = sqrt(max(1 - w*w, 0.0))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = q[1:]/s
    return axis, angle

# ----------------------------
# Student vs Solution versions
# ----------------------------
def student_compose(q1, q2):
    """TODO: return the composed quaternion (apply q2 then q1)."""
    return q_normalize(q_mul(q1, q2))


    raise NotImplementedError("Fill in student_compose()")

def solution_compose(q1, q2):
    return q_normalize(q_mul(q1, q2))

def student_slerp(q0, q1, u):
    """TODO: implement SLERP between q0 and q1 at fraction u in [0,1]."""

    # between 2 vectors, cos_theta = \frac{v0 \cdot v1}{||v0|| ||v1||}
    # in this case, both are unit quaternions, so denominator is 1
    # therefore, cos_theta = v0 \cdot v1
    # and so, \theta = \arccos(cos_theta)

    # The goal of SLERP is to interpolate along the great circle path on the 4D unit sphere
    # defined by the two quaternions. U can be thought of as the fraction of the angle between
    # the two quaternions that we want to rotate through.

    dot = np.dot(q0, q1)

    # If the dot product is negative (implies a projection of one vector onto the other is negative),
    # then we want to take the "short path" between the two quaternions.
    # This involves negating one of the quaternions (q1) to ensure the angle between them is less than 180 degrees.
    if dot < 0:
        dot = -dot
        q1 = -q1  # take the short path

    angle_between_quaternion = np.arccos(dot)

    # slerp general formula:
    # slerp(q0, q1, u) = (sin((1-u)*theta) / sin(theta)) * q0 + (sin(u*theta) / sin(theta)) * q1
    # note that this formula works for quaternions and also in 2d/3D vector spaces. 
    # its goal is to give a linear interpolation in terms of angle, not in terms of chord length.
    
    sin_theta = np.sin(angle_between_quaternion)
    # observe that if sin_theta is 0 (or close to 0 as in the case of very close quaternions),
    # we'll end up div0'ing. However, neatly, since sin(0) \approx theta for small theta,
    # we can just fall back to linear interpolation in that case. The linear interpolation
    # will be close enough to the great circle path for small angles.

    if sin_theta < 1e-6:
        return q_normalize((1-u)*q0 + u*q1)
    
    term0 = (np.sin((1-u)*angle_between_quaternion) / sin_theta) * q0
    term1 = (np.sin(u*angle_between_quaternion) / sin_theta) * q1

    return q_normalize(term0 + term1)


    # Hint: handle the "short path" (flip sign if dot<0); fall back to lerp if very close.
    # return q_slerp(q0, q1, u)
    raise NotImplementedError("Fill in student_slerp()")

def solution_slerp(q0, q1, u):
    return q_slerp(q0, q1, u)

def student_attitude_pd(q, q_des, omega_body, kp, kd):
    """TODO: compute body rates from quaternion error (PD in tangent space)."""
    # calculate the error quaternion
    # First, lets say q_des = q_e ⊗ q
    # Therefore, q_e = q_des ⊗ q*
    
    err_quat = q_mul(q_des, q_conj(q))

    # Standard quaternion representation is [w, x, y, z] which is [scalar, vector]
    # The vector part represents the axis of rotation, and the scalar part represents the angle of rotation.
    # At small angles, cos(theta/2) ~ 1, sin(theta/2) ~ theta/2
    # Therefore, for small error angles, we can approximate the error quaternion as:
    # q_e ~ [1, (theta/2)*u] where u is the unit rotation axis.
    # Resultingly, we can get the axis-angle error (axis-angle error represents the minimal rotation needed to correct the attitude error in the frame of the body)
    # axis_angle_error = 2 * vec(q_e) = 2 * (theta/2)*u = theta*u for small angles.
    axis_angle_error = 2.0 * err_quat[1:]
    p_term = kp * axis_angle_error

    # multiply by sign of scalar part to ensure shortest rotation.
    # recall that cos(theta/2) is the scalar part, and so if it's negative,
    # that implies theta/2 is in (90, 180] degrees, meaning theta is in (180, 360] degrees.
    # therefore, we want to flip the sign of the proportional term to ensure we take the short path.
    if err_quat[0] < 0:
        p_term = -p_term

    # The derivative term is kd (w_err), where w_err = w_des - w_body
    # Note that this is actually the whittled down formula
    # the true derivation for this comes from q_e_dot = 0.5 * q_e ⊗ [omega_des - R_e * omega_body], where R_e is the rotation matrix corresponding to q_e
    # But for small angles, R_e ~ I, which makes all of q_e_dot = 0.5 * q_e ⊗ [omega_des - omega_body]
    # As we can see, the derivative of the error quaternion is directly proportional to the difference in angular velocities, and so in practice
    # we can just use the body rates directly for the derivative term.
    d_term = kd * (-omega_body)

    omega_cmd = p_term + d_term
    return omega_cmd

    # Hint: error quaternion qe = q_des ⊗ q*
    # Small-angle approx: desired body rates omega_cmd ≈ kp * 2*vec(qe)*sign(qe.w) - kd * omega_body
    # return kp * 2*np.sign(qe[0]) * qe[1:] - kd * omega_body
    raise NotImplementedError("Fill in student_attitude_pd()")

def solution_attitude_pd(q, q_des, omega_body, kp, kd):
    qe = q_err(q_des, q)
    # ensure shortest rotation
    s = 1.0 if qe[0] >= 0 else -1.0
    omega_cmd = kp * (2.0*s*qe[1:]) - kd * omega_body
    return omega_cmd

# ----------------------------
# Plot helpers (single figure per task)
# ----------------------------
def plot_frame(ax, R, scale=1.0, label_prefix=""):
    o = np.zeros(3)
    axes = R * scale
    # draw three arrows for x, y, z
    for i, name in enumerate(["x", "y", "z"]):
        a = axes[:, i]
        ax.quiver(o[0], o[1], o[2], a[0], a[1], a[2])
        ax.text(a[0], a[1], a[2], f"{label_prefix}{name}")

def quat_to_R(q):
    """Return 3x3 rotation matrix for plotting frame axes."""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

def unit_axes():
    return np.eye(3)

# ----------------------------
# Exercises
# ----------------------------
def exercise_compose(args, use_student=False):
    # Build quaternions
    q1 = q_from_axis_angle(args.axis1, np.deg2rad(args.deg1))
    q2 = q_from_axis_angle(args.axis2, np.deg2rad(args.deg2))

    compose = student_compose if use_student else solution_compose
    qc = compose(q1, q2)  # apply q2 then q1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Composition: apply q2 then q1 (frames)")

    # World frame
    plot_frame(ax, unit_axes(), 1.0, label_prefix="W-")

    # q2 frame then q1∘q2
    R2 = quat_to_R(q2)
    plot_frame(ax, R2, 0.8, label_prefix="q2-")

    Rc = quat_to_R(qc)
    plot_frame(ax, Rc, 0.6, label_prefix="q1∘q2-")

    # also show applying in opposite order for contrast
    qo = solution_compose(q2, q1)
    Ro = quat_to_R(qo)
    plot_frame(ax, Ro, 0.6, label_prefix="q2∘q1-")

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    plt.show()

def exercise_slerp(args, use_student=False):
    q0 = q_from_euler_xyz(*np.deg2rad(args.q0_euler_deg))
    q1 = q_from_euler_xyz(*np.deg2rad(args.q1_euler_deg))
    slerp = student_slerp if use_student else solution_slerp

    # track path of body x-axis tip on unit sphere
    xs = []
    for i in range(args.steps+1):
        u = i/args.steps
        qi = slerp(q0, q1, u)
        tip = q_rotate(qi, np.array([1.0, 0.0, 0.0]))
        xs.append(tip)
    xs = np.array(xs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("SLERP path of body x-axis tip")
    # draw unit sphere (wireframe)
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(X, Y, Z, linewidth=0.3)

    # plot path
    ax.plot(xs[:,0], xs[:,1], xs[:,2], linewidth=2.0, marker='o')
    # show start/end frames
    plot_frame(ax, quat_to_R(q0), 0.5, "start-")
    plot_frame(ax, quat_to_R(q1), 0.5, "end-")

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1,1,1])
    plt.show()

def exercise_servo(args, use_student=False):
    q_des = q_from_euler_xyz(*np.deg2rad(args.q_des_euler_deg))
    q = q_from_euler_xyz(0.0, 0.0, 0.0)  # start at identity
    omega = np.zeros(3)

    ctrl = student_attitude_pd if use_student else solution_attitude_pd

    t = 0.0
    ts = []
    ang_err = []
    omega_log = []
    tip_log = []

    while t <= args.tfinal + 1e-9:
        # log
        qe = q_err(q_des, q)
        _, angle = axis_angle_from_q(qe)
        ts.append(t)
        ang_err.append(angle)
        omega_log.append(omega.copy())
        tip_log.append(q_rotate(q, np.array([1.0, 0.0, 0.0])))

        # control
        omega_cmd = ctrl(q, q_des, omega, args.kp, args.kd)

        # first-order rate servo to commanded omega (for a bit of realism)
        tau = 0.05
        domega = (omega_cmd - omega) / tau
        omega = omega + args.dt*domega

        # integrate attitude
        q = integrate_q(q, omega, args.dt)

        t += args.dt

    ts = np.array(ts)
    ang_err = np.array(ang_err)
    omega_log = np.array(omega_log)
    tip_log = np.array(tip_log)

    # Plot 1: angle error over time
    plt.figure()
    plt.title("Attitude error angle vs time")
    plt.plot(ts, np.rad2deg(ang_err))
    plt.xlabel("time [s]")
    plt.ylabel("angle error [deg]")
    plt.grid(True)

    # Plot 2: body rate magnitude over time
    plt.figure()
    plt.title("Body rate magnitude vs time")
    wmag = np.linalg.norm(omega_log, axis=1)
    plt.plot(ts, wmag)
    plt.xlabel("time [s]")
    plt.ylabel("|ω| [rad/s]")
    plt.grid(True)

    # Plot 3: path of body x-axis tip during convergence
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Tip of body x-axis during servo")
    ax.plot(tip_log[:,0], tip_log[:,1], tip_log[:,2])
    plot_frame(ax, quat_to_R(q_des), 0.6, label_prefix="desired-")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    plt.show()

# ----------------------------
# Argparse front-end
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Quaternion toy problems for roboticists")
    sub = p.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("compose", help="Demonstrate rotation composition and order")
    pc.add_argument("--deg1", type=float, default=90.0, help="Angle 1 (deg)")
    pc.add_argument("--axis1", type=float, nargs=3, default=[0,0,1], help="Axis 1 (x y z)")
    pc.add_argument("--deg2", type=float, default=90.0, help="Angle 2 (deg)")
    pc.add_argument("--axis2", type=float, nargs=3, default=[1,0,0], help="Axis 2 (x y z)")

    ps = sub.add_parser("slerp", help="Visualize SLERP path on unit sphere")
    ps.add_argument("--q0_euler_deg", type=float, nargs=3, default=[0,0,0], help="roll pitch yaw (deg) for q0")
    ps.add_argument("--q1_euler_deg", type=float, nargs=3, default=[0,90,0], help="roll pitch yaw (deg) for q1")
    ps.add_argument("--steps", type=int, default=25, help="Interpolation steps")

    pa = sub.add_parser("servo", help="PD attitude servo with quaternion error")
    pa.add_argument("--q_des_euler_deg", type=float, nargs=3, default=[0,45,0], help="Desired roll pitch yaw (deg)")
    pa.add_argument("--kp", type=float, default=3.0, help="Proportional gain on error angle")
    pa.add_argument("--kd", type=float, default=0.6, help="Derivative gain on body rates")
    pa.add_argument("--tfinal", type=float, default=3.0, help="Final time (s)")
    pa.add_argument("--dt", type=float, default=0.01, help="Time step (s)")

    p.add_argument("--use-student", action="store_true",
                   help="Use your student implementations (fill TODOs first!)")

    return p

def main():
    args = build_parser().parse_args()
    use_student = bool(args.use_student)
    if args.mode == "compose":
        exercise_compose(args, use_student)
    elif args.mode == "slerp":
        exercise_slerp(args, use_student)
    elif args.mode == "servo":
        exercise_servo(args, use_student)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
