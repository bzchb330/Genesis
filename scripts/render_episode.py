#!/usr/bin/env python
import argparse, imageio.v2 as imageio
from seqgrasp import SequentialGraspEnv
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="outputs/random_episode.mp4"); p.add_argument("--steps",type=int,default=300); p.add_argument("--checkpoint"); a=p.parse_args()
    from pathlib import Path; Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    env=SequentialGraspEnv(render_mode="rgb_array"); obs,_=env.reset(seed=0); env.action_space.seed(0); frames=[]
    policy=None
    if a.checkpoint:
        from stable_baselines3 import PPO
        policy=PPO.load(a.checkpoint)
    for _ in range(a.steps):
        action=env.action_space.sample() if policy is None else policy.predict(obs,deterministic=True)[0]
        obs,_,term,trunc,_=env.step(action); frames.append(env.render())
        if term or trunc: break
    imageio.mimsave(a.output,frames,fps=env.metadata["render_fps"]); env.close(); print(a.output)
if __name__=="__main__": main()
