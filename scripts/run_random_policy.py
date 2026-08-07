#!/usr/bin/env python
from seqgrasp import SequentialGraspEnv
def main():
    env=SequentialGraspEnv(); obs,info=env.reset(seed=0); env.action_space.seed(0)
    print("observation components:",[(m.name,m.dimension,m.unit) for m in env.observation_metadata])
    for i in range(1000):
        obs,reward,terminated,truncated,info=env.step(env.action_space.sample())
        if i%100==0: print(i,reward,info["reward_terms"])
        if terminated or truncated: obs,info=env.reset(seed=i+1)
    env.close(); print("completed steps:", i + 1)
if __name__=="__main__": main()
