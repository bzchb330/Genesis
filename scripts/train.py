#!/usr/bin/env python
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from seqgrasp import load_configs, SequentialGraspEnv
def main():
    cfg=load_configs(); Path(cfg.train.log_dir).mkdir(exist_ok=True); Path(cfg.train.checkpoint_dir).mkdir(exist_ok=True)
    env=make_vec_env(SequentialGraspEnv,n_envs=cfg.train.n_envs,seed=cfg.train.seed)
    model=PPO("MlpPolicy",env,seed=cfg.train.seed,tensorboard_log=cfg.train.log_dir,verbose=1)
    cb=CheckpointCallback(cfg.train.checkpoint_freq,cfg.train.checkpoint_dir); model.learn(cfg.train.total_timesteps,callback=cb)
if __name__=="__main__": main()

