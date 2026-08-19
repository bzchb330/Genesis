from __future__ import annotations

import numpy as np

from ..config import ROOT, load_configs
from ..control import resolve_hand_indices
from ..diagnostics.multi_grasp import load_grasp_profile
from ..diagnostics.scripted_grasp import _joint_target
from .phase2_5_trajectory import BAcquisitionTrajectory
from .resource_components import FINGER_ORDER


def trajectory_from_unit(cfg25, pose: dict, candidate_index: int, unit: np.ndarray) -> BAcquisitionTrajectory:
    cfg=load_configs(); from ..scene_builder import build_scene
    model,_=build_scene(cfg); indices=resolve_hand_indices(model,cfg.hand)
    _,profile=load_grasp_profile(ROOT/"configs"/"grasps"/"resource_grasp_A_02.yaml"); open_q=_joint_target(model,cfg,indices,profile.open_joint_fractions)
    anchors=open_q.copy()
    for finger,q in pose["representative_joint_rad"].items():
        i=FINGER_ORDER.index(finger); anchors[4*i:4*i+4]=q
    active=np.asarray([finger in pose["accessible_fingers"] for finger in FINGER_ORDER]); mask=np.repeat(active,4); cursor=0
    def target(width, base):
        nonlocal cursor
        result=base.copy(); offsets=np.interp(unit[cursor:cursor+16],[0,1],width); cursor+=16; result[mask]+=offsets[mask]; return result
    approach=target(cfg25.trajectory_search.joint_approach_offset_rad,anchors)
    pre=target(cfg25.trajectory_search.joint_precontact_offset_rad,anchors)
    closing=target(cfg25.trajectory_search.joint_closing_offset_rad,pre)
    hold=target(cfg25.trajectory_search.joint_hold_offset_rad,closing)
    close_steps=int(round(np.interp(unit[cursor],[0,1],cfg25.timing.close_steps_bounds))); cursor+=1
    delays=tuple(int(round(np.interp(value,[0,1],cfg25.trajectory_search.per_finger_close_delay_steps))) if active[i] else 0 for i,value in enumerate(unit[cursor:cursor+4])); cursor+=4
    release=int(round(np.interp(unit[cursor],[0,1],cfg25.timing.fixture_release_delay_steps_bounds)))
    ranges=model.jnt_range[indices.joint_ids]
    values=[np.clip(x,ranges[:,0],ranges[:,1]) for x in (approach,pre,closing,hold)]
    return BAcquisitionTrajectory(candidate_index,*(tuple(float(v) for v in x) for x in values),close_steps,delays,release)
