#!/usr/bin/env python
import argparse, collections
from stable_baselines3 import PPO
from seqgrasp import SequentialGraspEnv
def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--episodes",type=int,default=10); a=p.parse_args()
    model=PPO.load(a.checkpoint); env=SequentialGraspEnv(); successes=drops=0; failures=collections.Counter()
    for ep in range(a.episodes):
        obs,_=env.reset(seed=ep); done=False; last={}
        while not done: obs,_,term,trunc,last=env.step(model.predict(obs,deterministic=True)[0]); done=term or trunc
        successes += int(last.get("phase")==4 and not term); drops += int(last.get("failure_reason")=="object_a_dropped"); failures[last.get("phase")]+=int(term)
    # TODO(PI): success/drop thresholds are config-backed; disabled thresholds cannot be inferred here.
    print({"success_rate":successes/a.episodes,"drop_rate":drops/a.episodes,"per_phase_failures":dict(failures)})
if __name__=="__main__": main()

