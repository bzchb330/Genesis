import numpy as np
from seqgrasp import load_configs
from seqgrasp.control import ZeroRetentionController,resolve_hand_indices
from seqgrasp.env.resource import build_resource_state,compute_resource_metric
from seqgrasp.scene_builder import build_scene

def test_retention_and_resource_interfaces_expose_raw_state_without_defining_laws():
    cfg=load_configs(); model,data=build_scene(cfg); indices=resolve_hand_indices(model,cfg.hand)
    state=build_resource_state(model,data,cfg,0,indices); assert state.joint_positions.shape==(cfg.hand.dof_count,); assert state.joint_limits.shape==(cfg.hand.dof_count,2); assert set(state.object_poses)=={o.name for o in cfg.scene.objects}; assert compute_resource_metric(state,cfg) is None
    out=ZeroRetentionController().residual(state.joint_positions,state.joint_velocities,state.tactile_features,state.phase); np.testing.assert_array_equal(out,0)
