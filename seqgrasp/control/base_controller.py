import numpy as np

class JointImpedanceController:
    def __init__(self, stiffness, damping, torque_limit):
        self.kp, self.kd, self.limit = stiffness, damping, torque_limit
    def torque(self, desired_q, q, qvel):
        desired_q, q, qvel = map(lambda x: np.asarray(x, dtype=float), (desired_q, q, qvel))
        if desired_q.shape != q.shape or q.shape != qvel.shape: raise ValueError("controller vectors must have identical shapes")
        out = np.clip(self.kp*(desired_q-q)-self.kd*qvel, -self.limit, self.limit)
        if not np.all(np.isfinite(out)): raise ValueError("controller produced non-finite torque")
        return out
