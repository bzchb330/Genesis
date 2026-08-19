import numpy as np


def compute_phase2_tactile_features(
    contacts_by_finger,
    finger_order,
    binary_contact_threshold_N: float,
    zero_normal_epsilon_N: float,
) -> dict[str, np.ndarray]:
    """Return the three PI-specified Phase 2 tactile features per finger.

    The features are unnormalised: binary contact, summed normal force [N],
    and summed tangential force divided by summed normal force.  A finger with
    effectively zero normal force receives ratio zero; this encodes absence of
    a slip-proxy signal, not a loaded physical friction ratio of zero.
    """

    normal = np.asarray([
        sum(float(contact.normal_force) for contact in contacts_by_finger.get(finger, []))
        for finger in finger_order
    ], dtype=np.float64)
    tangential = np.asarray([
        sum(float(contact.tangential_force) for contact in contacts_by_finger.get(finger, []))
        for finger in finger_order
    ], dtype=np.float64)
    ratio = np.divide(
        tangential,
        normal,
        out=np.zeros_like(normal),
        where=normal > zero_normal_epsilon_N,
    )
    return {
        "binary_contact": (normal > binary_contact_threshold_N).astype(np.float32),
        "normal_force_N": normal,
        "tangential_to_normal_ratio": ratio,
    }

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
