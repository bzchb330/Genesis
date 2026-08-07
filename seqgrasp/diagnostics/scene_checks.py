from __future__ import annotations
import mujoco
import numpy as np
from ..scene_builder import build_scene, randomize_objects

def _half_height(obj): return obj.size[2] if obj.shape=="cube" else obj.size[1]

def check_initial_placements(cfg,seed=0):
    model,data=build_scene(cfg); randomize_objects(model,data,cfg,np.random.default_rng(seed))
    table_top=cfg.scene.table_pos[2]+cfg.scene.table_size[2]; objects=[]
    object_body_ids={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,o.name) for o in cfg.scene.objects}
    object_geom_ids={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f"{o.name}_geom") for o in cfg.scene.objects}
    hand_penetrations=[]
    for i in range(data.ncon):
        c=data.contact[i]; b1,b2=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
        if (c.geom1 in object_geom_ids and b2 not in object_body_ids and b2!=0) or (c.geom2 in object_geom_ids and b1 not in object_body_ids and b1!=0): hand_penetrations.append(i)
    for obj in cfg.scene.objects:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.name); pos=data.xpos[bid].copy()
        objects.append({"name":obj.name,"position":pos.tolist(),"inside_workspace":bool(np.all(pos>=cfg.scene.workspace_low) and np.all(pos<=cfg.scene.workspace_high)),"table_penetration":bool(pos[2]-_half_height(obj)<table_top-1e-12)})
    return {"seed":seed,"objects":objects,"hand_penetration":bool(hand_penetrations),"valid":all(o["inside_workspace"] and not o["table_penetration"] for o in objects) and not hand_penetrations}
