import numpy as np

def compute_tactile_features(contacts_by_finger, cfg) -> dict[str, np.ndarray]:
    """Return fixed-order flags [unitless] and total normal force [N] per finger."""
    fingers = list(cfg.hand.finger_geom_mapping)
    flags = np.asarray([bool(contacts_by_finger.get(f)) for f in fingers], dtype=np.float32)
    normal = np.asarray([sum(c.normal_force for c in contacts_by_finger.get(f, [])) for f in fingers], dtype=np.float32)
    if cfg.task.tactile_normalization is not None:
        normal /= cfg.task.tactile_normalization
    # TODO(PI): define physics of candidate features; placeholder is shaped zeros.
    extra = np.zeros(cfg.task.extra_tactile_feature_dim, dtype=np.float32)
    return {"contact_flags": flags, "normal_force": normal, "extra_pi_features": extra}
