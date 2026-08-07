from dataclasses import replace
from seqgrasp import load_configs,SequentialGraspEnv
from seqgrasp.env.observations import metadata,observation_spec

def test_observation_contract_and_privileged_toggle():
    cfg=load_configs(); spec=observation_spec(cfg); assert any(x["name"]=="privileged_target_position" and x["privileged"] and not x["enabled"] for x in spec)
    env=SequentialGraspEnv(cfg); obs,_=env.reset(seed=0); assert obs.size==sum(m.dimension for m in metadata(cfg)); env.close()
    flags=dict(cfg.task.observations); flags["privileged_target_position"]=True; enabled=replace(cfg,task=replace(cfg.task,observations=flags)); env=SequentialGraspEnv(enabled); obs,_=env.reset(seed=0); assert obs.size==sum(m.dimension for m in metadata(enabled)); assert obs.size==sum(m.dimension for m in metadata(cfg))+3; env.close()
