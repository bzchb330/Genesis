from enum import IntEnum
import numpy as np
import mujoco

class Phase(IntEnum): APPROACH_A=0; GRASP_A=1; APPROACH_B=2; GRASP_B=3; HOLD=4

def object_positions(model, data, scene_cfg):
    return {o.name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o.name)].copy() for o in scene_cfg.objects}

def update_phase(phase, tactile, cfg):
    # TODO(PI): define phase-transition conditions and use the config-backed
    # threshold placeholders. Until then, automatic transitions are disabled.
    return phase

def failure_reason(model, data, cfg, phase):
    for name, pos in object_positions(model, data, cfg.scene).items():
        if np.any(pos < cfg.scene.workspace_low) or np.any(pos > cfg.scene.workspace_high): return f"{name}_left_workspace"
    # TODO(PI): loss/drop semantics are disabled until drop_height_threshold is chosen.
    if cfg.task.drop_height_threshold is not None and phase >= Phase.APPROACH_B:
        if object_positions(model, data, cfg.scene)[cfg.scene.objects[0].name][2] < cfg.task.drop_height_threshold: return "object_a_dropped"
    return None
