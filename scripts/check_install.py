#!/usr/bin/env python
import os, platform
import gymnasium, mujoco
from gymnasium.utils.env_checker import check_env
from seqgrasp import load_configs, SequentialGraspEnv

def main():
    print("Python:",platform.python_version()); print("MuJoCo:",mujoco.__version__); print("Gymnasium:",gymnasium.__version__)
    print("MUJOCO_GL:",os.environ.get("MUJOCO_GL","default"))
    cfg=load_configs(); env=SequentialGraspEnv(cfg); obs,info=env.reset(seed=0); check_env(env,skip_render_check=True)
    print("Scene:",env.model.nbody,"bodies,",env.model.nu,"actuators"); print("Observation:",obs.shape,info)
    if os.environ.get("MUJOCO_GL") in {"egl","osmesa"}: print("Offscreen frame:",env.render().shape)
    env.close(); return 0
if __name__=="__main__": raise SystemExit(main())

