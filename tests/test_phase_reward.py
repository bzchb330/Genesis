from seqgrasp import load_configs,SequentialGraspEnv
from seqgrasp.env.rewards import compute_reward

def test_phase_and_placeholder_reward_instrumentation():
    cfg=load_configs(); env=SequentialGraspEnv(cfg); _,info=env.reset(seed=0); assert info["phase"]==0 and info["phase_transition_reason"] is None
    _,reward,_,_,info=env.step(env.action_space.sample()); assert info["phase"]==0 and info["phase_transition_reason"] is None and info["failure_reason"] is None
    assert set(info["reward_terms"])==set(cfg.task.reward_weights); assert reward==0.0
    total,terms=compute_reward({"action":env.action_space.sample(),"failure":None},cfg); assert total==0 and all(v==0 for v in terms.values()); env.close()
