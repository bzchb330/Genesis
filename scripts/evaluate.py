#!/usr/bin/env python
import argparse, collections
from seqgrasp import SequentialGraspEnv,load_configs
def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--episodes",type=int,default=10); a=p.parse_args()
    from stable_baselines3 import PPO
    cfg=load_configs(); model=PPO.load(a.checkpoint); env=SequentialGraspEnv(cfg); failures=collections.Counter(); raw=[]
    for ep in range(a.episodes):
        obs,_=env.reset(seed=ep); done=False; last={}
        while not done: obs,_,term,trunc,last=env.step(model.predict(obs,deterministic=True)[0]); done=term or trunc
        failures[last.get("phase")]+=int(term); raw.append({"seed":ep,"final_phase":last.get("phase"),"terminated":term,"truncated":trunc,"termination_reason":last.get("failure_reason")})
    # TODO(PI): define success/drop criteria. Until then rates are unavailable,
    # and this script reports neutral raw episode outcomes only.
    print({"success_rate":None,"drop_rate":None,"per_phase_terminations":dict(failures),"episodes":raw})
if __name__=="__main__": main()
