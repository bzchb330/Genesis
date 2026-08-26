from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw
import mujoco
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3.control import ContactAwareCloser, actuator_target_from_qpos
from seqgrasp.phase3c06 import (
    _project,
    build_sphere_scene,
    construct_palmodigital_pockets,
    deterministic_acquisition_position,
    load_acquisition_states,
    load_phase3c06_config,
    restore_acquisition_state,
    run_storage_trial,
)
from seqgrasp.phase3.model import set_fixture, set_object_pose


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.35, -0.02, 0.02]
    camera.distance = 0.45
    camera.azimuth = 135
    camera.elevation = -20
    return camera


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(632, 18 + 7 * len(label)), 34), fill=(0, 0, 0))
    draw.text((14, 14), label, fill=(255, 255, 255))
    return np.asarray(image)


def _write(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                            macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


def acquisition_frames() -> list[np.ndarray]:
    scene = build_sphere_scene(); renderer = mujoco.Renderer(scene.model, 480, 640); camera = _camera()
    open_qpos, pre_qpos, pinch_qpos = (_project(scene, name) for name in ("open hand", "pre grasp", "two finger pinch"))
    open_target, pre_target, pinch_target = (
        actuator_target_from_qpos(scene, value) for value in (open_qpos, pre_qpos, pinch_qpos)
    )
    scene.data.qpos[:24] = open_qpos; scene.data.ctrl[:] = open_target
    set_object_pose(scene, deterministic_acquisition_position(scene, 0)); set_fixture(scene, True)
    mujoco.mj_forward(scene.model, scene.data)
    frames=[]
    def record(label):
        renderer.update_scene(scene.data, camera=camera); frames.append(_annotate(renderer.render(), label))
    record("OPEN_HAND - sphere fixture active")
    ids=np.r_[scene.actuator_ids["wrist"],scene.actuator_ids["thumb"],scene.actuator_ids["index"]]
    for step in range(80):
        alpha=(step+1)/80; scene.data.ctrl[ids]=(1-alpha)*open_target[ids]+alpha*pre_target[ids];mujoco.mj_step(scene.model,scene.data)
        if step%5==0: record("thumb/index approach - fixture active")
    closer=ContactAwareCloser(scene,float(load_phase3c06_config()["diagnostic"]["contact_force_n"])); ids=np.r_[scene.actuator_ids["thumb"],scene.actuator_ids["index"]]
    for step in range(180):
        alpha=(step+1)/180; proposed=scene.data.ctrl.copy();proposed[ids]=(1-alpha)*pre_target[ids]+alpha*pinch_target[ids];scene.data.ctrl[:]=closer.limit_target(proposed);mujoco.mj_step(scene.model,scene.data)
        if step%5==0: record("thumb/index contact-aware acquisition")
    for step in range(50):
        mujoco.mj_step(scene.model,scene.data)
        if step%5==0: record("dual-contact acquired state - fixture active")
    renderer.close(); return frames


def storage_frames(state_index: int, pocket_name: str, preshape: str, wrist: tuple[float,float]) -> tuple[list[np.ndarray], dict]:
    scene=build_sphere_scene();state=load_acquisition_states()[state_index];pocket=construct_palmodigital_pockets(scene)[pocket_name]
    renderer=mujoco.Renderer(scene.model,480,640);camera=_camera();frames=[];stages=[]
    def callback(current,step,stage):
        if step%5==0 or stage!=stages[-1] if stages else True:
            renderer.update_scene(current.data,camera=camera)
            frames.append(_annotate(renderer.render(),f"{pocket_name} | {preshape} | wrist {wrist} | {stage}"));stages.append(stage)
    result=run_storage_trial(scene,state,pocket,preshape,wrist,frame_callback=callback)
    renderer.close();return frames,result


def main() -> None:
    output=ROOT/"outputs/phase3C06/videos"; output.mkdir(parents=True,exist_ok=True)
    _write(output/"d0_thumb_index_acquisition.mp4",acquisition_frames())
    frames,_=storage_frames(0,"middle_ring","NO_PRESHAPE",(0.0,0.0));_write(output/"open_corridor_sphere_transfer.mp4",frames)
    frames,_=storage_frames(0,"ring_little","PRESHAPE",(5.0,-5.0));_write(output/"ring_little_pocket_failure.mp4",frames)
    frames,result=storage_frames(23,"old_palm_center","PRESHAPE",(5.0,5.0))
    capture_frames=frames[:max(1,min(len(frames),62))]
    release_frames=frames[max(0,min(len(frames)-1,55)):]
    _write(output/"preshape_capture_and_wrist_settling.mp4",capture_frames)
    _write(output/"palm_center_thumb_release_failure.mp4",release_frames)
    print({"videos":[str(path) for path in sorted(output.glob("*.mp4"))],
           "truth_note":"All frames are actual MuJoCo replays; no successful recovery/retention video exists.",
           "representative_result":{"stable_capture":result["stable_capture"],"thumb_recovered":result["thumb_recovered"]}})


if __name__ == "__main__":
    main()
