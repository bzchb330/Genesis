def compute_reward(state, cfg):
    """Reward plumbing with PI-owned scientific terms left as placeholders."""
    terms = {"retention": 0.0, "progress": 0.0, "resource_j": 0.0, "regularization": 0.0, "failure": 0.0}
    # TODO(PI): define retention, phase progress, resource J,
    # action/energy regularization, and failure metrics.
    total = sum(cfg.task.reward_weights[k] * v for k, v in terms.items())
    return float(total), terms
