import numpy as np
from seqgrasp import SequentialGraspEnv
def rollout(seed,actions):
    env=SequentialGraspEnv(); out=[env.reset(seed=seed)[0].copy()]
    for a in actions: out.append(env.step(a)[0].copy())
    env.close(); return np.asarray(out)
def test_fixed_seed_actions_are_deterministic():
    actions=np.random.default_rng(3).uniform(-1,1,(20,16)).astype(np.float32)
    np.testing.assert_array_equal(rollout(7,actions),rollout(7,actions))

