"""Saved-state videos only; no live dynamics, receiver, release or success claim."""
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from seqgrasp import phase3cp05r as p


def main():
    summary=p.read('summary.json'); first=p.config()['candidates'][0]; snap=summary['equilibrium_audit'][0]['snapshot']
    clips=[dict(name='hand_startup_before_reset_fix.mp4',physics=first,
                source=p.read('equilibrium_states/'+first+'.json')['history'],limit=501,hand_only=True,stride=10,label='Nominal hand-only startup; not equilibrium'),
           dict(name='hand_settled_equilibrium.mp4',physics=first,source=snap['restore_confirmation_trace'],limit=None,hand_only=True,stride=5,label='Cached equilibrium; ORIGINAL targets maintained')]
    for name,short in zip(p.config()['candidates'],['IMP99','TC10IMP99']):
        trial=next(t for t in summary['trials'] if t['physics_name']==name and t['nominal_dt_s']==.002)
        clips.append(dict(name=f'MRL_{short}_repaired_diagnostic.mp4',physics=name,source=trial['trace'],limit=None,hand_only=False,stride=10,label='Fixed sphere: sustained R/L only; middle absent; no V1'))
    dest=p.OUTPUT/'videos'; dest.mkdir(parents=True,exist_ok=True); results=[]
    for clip in clips:
        rows=p.old.load_trace(clip['source']); rows=rows[:clip['limit']] if clip['limit'] else rows
        scene=p.old.setup_hand(clip['physics'],.002)
        if clip['hand_only']: scene.model.geom_rgba[p.old.native.a._object_geom_id(scene),3]=0
        renderer=mujoco.Renderer(scene.model,480,640); camera=mujoco.MjvCamera()
        camera.azimuth=135; camera.elevation=-18; camera.distance=.34; camera.lookat[:]=[.37,.015,.045]
        chosen=list(range(0,len(rows),clip['stride'])); slowdown=1/(.002*clip['stride']*25)
        with imageio.get_writer(dest/clip['name'],fps=25,codec='libx264',quality=7,macro_block_size=None) as writer:
            for number,i in enumerate(chosen):
                row=rows[i]; scene.data.qpos[:]=row['qpos']; scene.data.qvel[:]=row['qvel']; scene.data.ctrl[:]=row['ctrl']; mujoco.mj_forward(scene.model,scene.data)
                renderer.update_scene(scene.data,camera=camera); frame=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,640,96),fill='black')
                txt=[clip['physics'],clip['label'],f"recorded elapsed={row['time_s']-rows[0]['time_s']:.3f}s | {slowdown:g}x slow motion | saved-state replay",
                     'Sphere collision-disabled/hidden' if clip['hand_only'] else f"Fn={row['total_normal_force_n']:.4f} N | overlap={row['maximum_penetration_m']*1000:.4f} mm | WELD ON"]
                for k,line in enumerate(txt): draw.text((8,8+k*21),line,fill='white')
                writer.append_data(np.asarray(frame))
                if number in (0,len(chosen)//2,len(chosen)-1): frame.save(dest/(clip['name'].replace('.mp4',f'_{number}.png')))
        renderer.close(); results.append(dict(physics_name=clip['physics'],path=(dest/clip['name']).relative_to(p.ROOT).as_posix(),source=clip['source'],frames=len(chosen),fps=25,slowdown=slowdown))
    p.save('videos.json',dict(physics_names=p.config()['candidates'],generated=results,physics_steps=0)); print('Generated',len(results),'saved-state videos')


if __name__=='__main__': main()
