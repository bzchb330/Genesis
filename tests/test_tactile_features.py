from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from seqgrasp import load_configs
from seqgrasp.sensing.tactile_features import compute_tactile_features

def test_reference_tactile_features_order_zero_repeatability_and_normalization():
    cfg=load_configs(); fingers=list(cfg.hand.finger_geom_mapping); empty={f:[] for f in fingers}
    zero=compute_tactile_features(empty,cfg)
    assert zero["contact_flags"].shape==(len(fingers),); assert zero["normal_force"].shape==(len(fingers),)
    np.testing.assert_array_equal(zero["contact_flags"],0); np.testing.assert_array_equal(zero["normal_force"],0)
    contacts={f:[] for f in fingers}; contacts[fingers[1]]=[SimpleNamespace(normal_force=2.5),SimpleNamespace(normal_force=1.5)]
    first=compute_tactile_features(contacts,cfg); second=compute_tactile_features(contacts,cfg)
    np.testing.assert_array_equal(first["contact_flags"],second["contact_flags"]); np.testing.assert_array_equal(first["normal_force"],second["normal_force"])
    expected_flags=np.zeros(len(fingers)); expected_force=np.zeros(len(fingers)); expected_flags[1]=1; expected_force[1]=4
    np.testing.assert_array_equal(first["contact_flags"],expected_flags); np.testing.assert_array_equal(first["normal_force"],expected_force)
    scaled=replace(cfg,task=replace(cfg.task,tactile_normalization=2.0)); assert compute_tactile_features(contacts,scaled)["normal_force"][1]==2.0
