"""Render saved axisymmetric bench trajectories only; zero physics steps."""
import json
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from seqgrasp import phase3cp0 as p, contact_bench as b


def main():
    dest=p.OUTPUT/'videos'; dest.mkdir(parents=True,exist_ok=True)
    path=dest/'sphere_plane_contact_bench.mp4'
    snapshot=p.read('current_physics.json'); scene=b.create_bench(snapshot)
    renderer=mujoco.Renderer(scene.model,480,640)
    camera=mujoco.MjvCamera(); camera.distance=.11; camera.azimuth=90; camera.elevation=-15; camera.lookat[:]=[0,0,.0125]
    clips=[(p.config()['legacy_name'],'LEGACY: 0.50 N load-unload'),('OPTION_TC10_IMP99','UNAPPROVED OPTION TC10+IMP99: 0.50 N cycle')]
    count=0
    with imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=None) as writer:
        for name,label in clips:
            run=b.load_trace(p.OUTPUT/name/'cycle_03.npz')
            for i in range(0,len(run['samples']),20):
                row=dict(zip(run['fields'],run['samples'][i]))
                scene.data.qpos[2]=row['z_m']; scene.data.qvel[2]=row['vz_mps']
                mujoco.mj_forward(scene.model,scene.data)
                renderer.update_scene(scene.data,camera=camera)
                frame=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,640,77),fill=(0,0,0))
                draw.text((8,7),label,fill='white')
                draw.text((8,26),f"Recorded t={row['time_s']:.3f}s | load {row['target_load_n']:.4f} N | Fn {row['normal_force_n']:.4f} N",fill='white')
                draw.text((8,45),f"Signed overlap {row['signed_overlap_m']*1000:.4f} mm | negative = separated",fill='white')
                draw.text((8,61),'Saved-state rendering at 1x time; no hand or grasp-success claim',fill='white')
                writer.append_data(np.asarray(frame)); count+=1
                if i==0: frame.save(dest/(name+'_preview.png'))
            frame.save(dest/(name+'_last.png'))
    renderer.close()
    meta=dict(generated=[path.relative_to(p.ROOT).as_posix()],frames=count,fps=25,duration_s=count/25,
              physics_steps_during_rendering=0,reconstruction='Recorded z trajectory in exactly axial sphere-plane geometry; no trajectory extrapolation.',
              fixed_sphere_force_control_video=None,reason='Hand force-control stage not approved/executed')
    p.save('videos.json',meta); print(json.dumps(meta))


if __name__=='__main__': main()
