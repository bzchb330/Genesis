import numpy as np

def residual_torque(base, action, scale, limit, torque_limit):
    base, action = np.asarray(base, dtype=float), np.asarray(action, dtype=float)
    if base.shape != action.shape: raise ValueError("base and residual action dimensions differ")
    residual = np.clip(action*scale, -limit, limit)
    out = np.clip(base + residual, -torque_limit, torque_limit)
    if not np.all(np.isfinite(out)): raise ValueError("residual controller produced non-finite torque")
    return out
