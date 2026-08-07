from gymnasium.utils.env_checker import check_env
from seqgrasp import SequentialGraspEnv
def test_env_api_and_observation_dimension():
    env=SequentialGraspEnv(); check_env(env,skip_render_check=True)
    obs,_=env.reset(seed=1); assert obs.size==sum(x.dimension for x in env.observation_metadata)

