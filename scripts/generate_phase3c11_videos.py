"""Render only the one actually executed Phase 3C-1.1 preload hold."""
from __future__ import annotations
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from seqgrasp.config import ROOT
from seqgrasp.phase3c08 import build_forearm_scene

OUTPUT=ROOT/"outputs/phase3C11"

def annotate(frame,text):
    im=Image.fromarray(frame); draw=ImageDraw.Draw(im); draw.rectangle((8,8,632,40),fill=(0,0,0)); draw.text((14,15),text,fill=(255,255,255)); return np.asarray(im)

def render(series_path:Path,destination:Path,label:str,stop:int,stride:int):
    scene=build_forearm_scene(with_actuator=True).scene; data=np.load(series_path,allow_pickle=False)
    camera=mujoco.MjvCamera(); camera.lookat[:]=[.35,-.02,.02]; camera.distance=.42; camera.azimuth=135; camera.elevation=-18
    renderer=mujoco.Renderer(scene.model,480,640); destination.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(destination,fps=20,codec="libx264",quality=7,macro_block_size=None) as writer:
        for i in range(0,min(stop,len(data["qpos"])),stride):
            scene.data.qpos[:]=data["qpos"][i]; mujoco.mj_forward(scene.model,scene.data); renderer.update_scene(scene.data,camera=camera)
            writer.append_data(annotate(renderer.render(),f"{label} | executed step {i+1} | B03={bool(data['inside_B03'][i])}"))
    renderer.close()

def main():
    result=json.loads((OUTPUT/"preloaded_B03_results.json").read_text()); executed=[r for r in result["rows"] if r.get("timeseries_path")]; videos=OUTPUT/"videos"; generated=[]
    if executed:
        row=executed[0]; source=Path(row["timeseries_path"]); generated=[videos/"representative_preloaded_B03_recheck.mp4",videos/"representative_sphere_failure.mp4"]
        render(source,generated[0],f"preloaded B03 recheck; {result['classification']}",1000,10)
        render(source,generated[1],f"sphere failure; first contact loss {row['first_contact_loss_step']}",100,2)
    summary_path=OUTPUT/"phase3c11_summary.json"; summary=json.loads(summary_path.read_text()); summary["videos"]=[str(p) for p in generated]; summary["video_omissions"]={"successful_preloaded_B03_hold":"no success occurred","cube_stable_hold":"no executable initializer","cylinder_stable_hold":"no executable initializer","thumb_assisted_storage_hold":"no robust ROLE-T candidate","handoff":"prohibited and not run"}; summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps({"generated":[str(p) for p in generated],"dynamics_replays":0,"executed_protocol_trials":len(executed),"fabricated_success_videos":0},indent=2))

if __name__=="__main__": main()
