#!/usr/bin/env python
import argparse, json, warnings
from pathlib import Path
import mujoco
import numpy as np
from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene, randomize_objects
from seqgrasp.diagnostics import check_initial_placements

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=0); p.add_argument("--render"); args=p.parse_args(); cfg=load_configs()
    report=check_initial_placements(cfg,args.seed); print(json.dumps(report,indent=2))
    if args.render:
        try:
            model,data=build_scene(cfg); randomize_objects(model,data,cfg,np.random.default_rng(args.seed))
            with mujoco.Renderer(model,cfg.scene.render_height,cfg.scene.render_width) as renderer:
                renderer.update_scene(data); import imageio.v2 as imageio; Path(args.render).parent.mkdir(parents=True,exist_ok=True); imageio.imwrite(args.render,renderer.render())
        except Exception as exc: warnings.warn(f"placement render unavailable: {exc}")
    return 0 if report["valid"] else 1
if __name__=="__main__": raise SystemExit(main())
