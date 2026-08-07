from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from ..config import load_configs
from ..scene_builder import build_scene, randomize_objects
from ..control import JointImpedanceController, hand_state, residual_torque, resolve_hand_indices
from .observations import build_observation, metadata
from .termination import Phase, update_phase, failure_reason
from .rewards import compute_reward

class SequentialGraspEnv(gym.Env):
    metadata={"render_modes":["rgb_array"]}
    def __init__(self,cfg=None,render_mode=None):
        self.cfg=cfg or load_configs(); self.model,self.data=build_scene(self.cfg); self.render_mode=render_mode
        self.metadata={"render_modes":["rgb_array"],"render_fps":round(1/(self.cfg.scene.timestep*self.cfg.scene.frame_skip))}
        n=self.cfg.hand.dof_count; self.action_space=spaces.Box(-1,1,(n,),np.float32)
        dim=sum(x.dimension for x in metadata(self.cfg)); self.observation_space=spaces.Box(-np.inf,np.inf,(dim,),np.float32)
        self.controller=JointImpedanceController(self.cfg.task.impedance_stiffness,self.cfg.task.impedance_damping,self.cfg.task.torque_limit)
        self.indices=resolve_hand_indices(self.model,self.cfg.hand)
        self._renderer=None; self.phase=Phase.APPROACH_A; self.steps=0; self.desired_q=np.zeros(n)
    @property
    def observation_metadata(self): return metadata(self.cfg)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed); mujoco.mj_resetData(self.model,self.data); randomize_objects(self.model,self.data,self.cfg,self.np_random)
        self.phase=Phase.APPROACH_A; self.steps=0; self.desired_q=hand_state(self.data,self.indices)[0]
        obs,_=build_observation(self.model,self.data,self.cfg,self.phase,self.indices)
        return obs,{"phase":int(self.phase),"phase_name":self.phase.name,"phase_transition_reason":None,"failure_reason":None}
    def step(self,action):
        action=np.asarray(action,dtype=np.float64); n=self.cfg.hand.dof_count
        if action.shape != (n,): raise ValueError(f"action must have shape {(n,)}")
        q,qvel=hand_state(self.data,self.indices)
        base=self.controller.torque(self.desired_q,q,qvel)
        torque=action*self.cfg.task.torque_limit if self.cfg.task.control_mode=="direct_torque" else residual_torque(base,action,self.cfg.task.residual_scale,self.cfg.task.residual_limit,self.cfg.task.torque_limit)
        self.data.ctrl[self.indices.actuator_ids]=torque
        for _ in range(self.cfg.scene.frame_skip): mujoco.mj_step(self.model,self.data)
        self.steps+=1; obs,tactile=build_observation(self.model,self.data,self.cfg,self.phase,self.indices)
        new_phase,transition_reason=update_phase(self.phase,tactile,self.cfg)
        if new_phase != self.phase: obs,_=build_observation(self.model,self.data,self.cfg,new_phase,self.indices)
        self.phase=new_phase
        reason=failure_reason(self.model,self.data,self.cfg,self.phase); terminated=reason is not None; truncated=self.steps>=self.cfg.task.episode_steps
        reward,terms=compute_reward({"action":action,"failure":reason},self.cfg)
        return obs,reward,terminated,truncated,{"phase":int(self.phase),"phase_name":self.phase.name,"phase_transition_reason":transition_reason,"reward_terms":terms,"failure_reason":reason}
    def render(self):
        if self._renderer is None: self._renderer=mujoco.Renderer(self.model,self.cfg.scene.render_height,self.cfg.scene.render_width)
        self._renderer.update_scene(self.data); return self._renderer.render()
    def close(self):
        if self._renderer is not None: self._renderer.close(); self._renderer=None
