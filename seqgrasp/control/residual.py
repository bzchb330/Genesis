import numpy as np

def residual_torque(base, action, scale, limit, torque_limit):
    residual = np.clip(np.asarray(action)*scale, -limit, limit)
    return np.clip(base + residual, -torque_limit, torque_limit)

