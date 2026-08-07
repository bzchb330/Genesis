import mujoco
import numpy as np
from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene,randomize_objects
from seqgrasp.diagnostics import check_initial_placements

def _positions(cfg,seed):
    model,data=build_scene(cfg); randomize_objects(model,data,cfg,np.random.default_rng(seed)); return np.asarray([data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,o.name)].copy() for o in cfg.scene.objects])

def test_initial_placement_valid_and_seeded():
    cfg=load_configs(); assert check_initial_placements(cfg,4)["valid"]
    np.testing.assert_array_equal(_positions(cfg,4),_positions(cfg,4)); assert not np.array_equal(_positions(cfg,4),_positions(cfg,5))
