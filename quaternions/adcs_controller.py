import numpy as np

def q_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def q_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

class ADCSController:
    def __init__(self, kp=2.0, kd=0.6):
        self.kp = kp
        self.kd = kd
        pass

    def step(self, q_desired, omega_current, q_current):
        q_err  = q_mult(q_conj(q_current), q_desired)
        p_term = self.kp * 2 * q_err[1:] * np.sign(q_err[0])
        d_term = self.kd * (-omega_current)
        omega_cmd = p_term + d_term
        return omega_cmd