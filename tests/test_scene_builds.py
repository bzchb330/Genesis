from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene
def test_scene_builds():
    cfg=load_configs(); model,data=build_scene(cfg); assert model.nu==cfg.hand.dof_count and len(cfg.scene.objects)==2
