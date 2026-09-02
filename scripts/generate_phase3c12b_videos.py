"""Render saved executed states only. No dynamics, fabricated release, or success."""
import json
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from seqgrasp import phase3c12b as b


def render(clips,filename):
    cfg=b.config().video; scene=b.build(); renderer=mujoco.Renderer(scene.model,cfg['height'],cfg['width'])
    camera=mujoco.MjvCamera(); camera.distance=cfg['distance_m']; camera.azimuth=cfg['azimuth']; camera.elevation=cfg['elevation']
    dest=b.OUTPUT/'videos'/filename; dest.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(dest,fps=cfg['fps'],codec='libx264',quality=7,macro_block_size=None) as writer:
        for path,label in clips:
            rows=b.load_series(path)
            for i in range(0,len(rows),cfg['stride']):
                row=rows[i]; scene.data.qpos[:]=row['qpos']; mujoco.mj_forward(scene.model,scene.data)
                camera.lookat[:]=np.asarray(row['sphere_position_world_m'])+[0,0,-.01]
                renderer.update_scene(scene.data,camera=camera); frame=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,cfg['width'],65),fill=(0,0,0))
                draw.text((8,8),label,fill='white'); draw.text((8,27),f"Executed step {row['step']} | sim t={row['time_s']:.3f}s | WELD ON | no release",fill='white')
                draw.text((8,46),f"Overlap {row['maximum_penetration_m']*1000:.3f} mm | replay at 0.25x simulation time",fill='white')
                writer.append_data(np.asarray(frame))
                if i==0: frame.save(dest.with_suffix('.preview.png'))
    renderer.close(); return dest.relative_to(b.ROOT).as_posix()


if __name__=='__main__':
    summary=b.read('phase3c12b_summary.json'); selections={r['surface']:r for r in summary['selected_primitives']}
    clips=[(selections[s]['timeseries'],f'Fixed-support {s}: virtual offset {selections[s]["virtual_offset"]}') for s in ('little','middle','ring')]
    videos=[render(clips,'fixed_sphere_preload_debug.mp4'),render([(b.read('receiver_construction.json')['timeseries'],'Welded MRL construction: invalid receiver; no release')],'welded_MRL_receiver_construction.mp4')]
    b.save('videos.json',dict(generated=videos,omitted={'weld_release_receiver_trial.mp4':'No release executed','validated_MRL_receiver_1000step.mp4':'No validated receiver','MRL_receiver_release_failure.mp4':'No post-release failure exists'},physics_steps_during_rendering=0))
    print(json.dumps(videos))
