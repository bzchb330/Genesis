import numpy as np
from seqgrasp import load_configs
from seqgrasp.control import ZeroRetentionController,resolve_hand_indices
from seqgrasp.env.resource import build_resource_state,compute_resource_metric
from seqgrasp.env.grasp_criteria import GraspCriterionState,is_grasp_acquired,is_object_lost,is_object_retained
from seqgrasp.scene_builder import build_scene

def test_retention_and_resource_interfaces_expose_raw_state_without_defining_laws():
    cfg=load_configs(); model,data=build_scene(cfg); indices=resolve_hand_indices(model,cfg.hand)
    state=build_resource_state(model,data,cfg,0,indices); assert state.joint_positions.shape==(cfg.hand.dof_count,); assert state.joint_limits.shape==(cfg.hand.dof_count,2); assert set(state.object_poses)=={o.name for o in cfg.scene.objects}; assert compute_resource_metric(state,cfg) is None
    out=ZeroRetentionController().residual(state.joint_positions,state.joint_velocities,state.tactile_features,state.phase); np.testing.assert_array_equal(out,0)
    criterion=GraspCriterionState(state.object_poses[cfg.diagnostic.object_name],np.zeros(6),0.0,np.zeros(len(cfg.hand.finger_geom_mapping)),state.tactile_features,0.0,False)
    assert is_grasp_acquired(criterion,cfg) is None; assert is_object_retained(criterion,cfg) is None; assert is_object_lost(criterion,cfg) is None
