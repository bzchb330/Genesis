"""Truthful saved-state impact clips; no hand video from millisecond censored starts."""
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from seqgrasp import phase3cp05 as p, contact_physics as c, contact_bench as b


def main():
    dest=p.OUTPUT/'videos'; dest.mkdir(parents=True,exist_ok=True); generated=[]
    s=p.read(p.OUTPUT/'summary.json')
    for n in p.config()['candidates']:
        label=n.replace('PHYSICS_CANDIDATE_',''); fname='dynamic_contact_'+label.replace('TC10_IMP99','TC10IMP99')+'.mp4'
        x=next(x for x in s['impact'] if x['physics_name']==n and x['dt_s']==.002 and x['height_m']==.005)
        rows=p.load_trace(x['trace']); scene=b.create_bench(p.read(p.ROOT/'outputs/phase3CP0/current_physics.json'),c.version(n))
        renderer=mujoco.Renderer(scene.model,480,640); camera=mujoco.MjvCamera()
        camera.distance=.075; camera.azimuth=90; camera.elevation=-10; camera.lookat[:]=[0,0,.014]
        # 0.004 s recorded increment at 25 fps => explicit 10x slow motion.
        with imageio.get_writer(dest/fname,fps=25,codec='libx264',quality=7,macro_block_size=None) as writer:
            for i in range(0,len(rows),2):
                r=rows[i]; scene.data.qpos[:]=r['qpos']; scene.data.qvel[:]=r['qvel']; mujoco.mj_forward(scene.model,scene.data)
                renderer.update_scene(scene.data,camera=camera); frame=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,640,95),fill='black')
                for k,text in enumerate([n,'5 mm drop | dt 2 ms | saved-state replay | 10x slow motion',
                    f"t={r['time_s']:.3f} s | overlap={r['penetration_m']*1000:.4f} mm | Fn={r['normal_force_n']:.4f} N",
                    'DIAGNOSTIC ONLY - no production physics selected']): draw.text((8,8+21*k),text,fill='white')
                writer.append_data(np.asarray(frame))
                if i in (0,50,500): frame.save(dest/f'{label}_{i}.png')
        renderer.close(); generated.append(dict(physics_name=n,path=(dest/fname).relative_to(p.ROOT).as_posix(),frames=len(rows[::2]),fps=25,slowdown=10,source=x['trace']))
    p.save('videos.json',dict(physics_names=p.config()['candidates'],generated=generated,physics_steps=0,
        hand_videos_generated=False,hand_video_reason='Only 1-3 dynamics steps before a force stop; a hand motion video would imply nonexistent sustained evidence. Static geometry and startup plots provided instead.'))
    print('Generated',len(generated),'saved-state diagnostic videos; no physics steps')


if __name__=='__main__': main()
