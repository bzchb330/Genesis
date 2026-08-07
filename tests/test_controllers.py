import numpy as np
import pytest
from seqgrasp.control import JointImpedanceController, residual_torque
from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene
from seqgrasp.control import resolve_hand_indices

def test_impedance_stiffness_damping_saturation_dimensions_and_determinism():
    controller=JointImpedanceController(stiffness=2.0,damping=.5,torque_limit=1.0)
    desired=np.array([1.,-.5,0.]); q=np.array([0.,0.,0.]); qvel=np.array([0.,1.,-1.])
    expected=np.clip(2*(desired-q)-.5*qvel,-1,1)
    np.testing.assert_array_equal(controller.torque(desired,q,qvel),expected)
    np.testing.assert_array_equal(controller.torque(desired,q,qvel),controller.torque(desired,q,qvel))
    assert np.all(np.isfinite(expected)); assert np.max(np.abs(expected))<=1
    with pytest.raises(ValueError): controller.torque(desired,q[:2],qvel)
    cfg=load_configs(); model,_=build_scene(cfg); indices=resolve_hand_indices(model,cfg.hand); upper=model.jnt_range[indices.joint_ids,1]
    near_limit=controller.torque(upper+.1,upper,np.zeros_like(upper)); assert near_limit.shape==(cfg.hand.dof_count,); assert np.all(np.isfinite(near_limit)); assert np.max(np.abs(near_limit))<=1

def test_residual_zero_limits_safety_and_dimensions():
    base=np.array([.2,-.3]); np.testing.assert_array_equal(residual_torque(base,np.zeros(2),.2,.1,1.),base)
    out=residual_torque(base,np.array([1e9,-1e9]),.2,.1,.25); np.testing.assert_allclose(out,[.25,-.25]); assert np.max(np.abs(out))<=.25
    with pytest.raises(ValueError): residual_torque(base,np.zeros(3),.2,.1,1.)
