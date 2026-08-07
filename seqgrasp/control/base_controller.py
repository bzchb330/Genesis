import numpy as np

class JointImpedanceController:
    def __init__(self, stiffness, damping, torque_limit): self.kp, self.kd, self.limit = stiffness, damping, torque_limit
    def torque(self, desired_q, q, qvel): return np.clip(self.kp*(desired_q-q)-self.kd*qvel, -self.limit, self.limit)

