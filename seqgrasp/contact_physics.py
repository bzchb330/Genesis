"""Explicit versioned hand-object pairs; legacy/env/self physics remain intact."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from .config import ROOT
from .phase3.model import build_shadow_scene, _name_runtime_collision_geoms
from .phase3c07 import phase3c07_scene_config
from .phase3c08 import _forearm_transform

REGISTRY=ROOT/'configs/contact_physics_registry.yaml'
LEGACY='LEGACY_PHASE3C_CONTACT_PHYSICS'
IMP99='PHYSICS_CANDIDATE_IMP99'
TC10='PHYSICS_CANDIDATE_TC10_IMP99'


def registry(): return yaml.safe_load(REGISTRY.read_text())
def version(name):
    r=registry()
    if name=='PHYSICS_V1_NEAR_RIGID':
        path=ROOT/'configs/PHYSICS_V1_NEAR_RIGID.yaml'
        if not path.exists(): raise PermissionError('No selected/frozen V1 exists')
        v=yaml.safe_load(path.read_text())
        if not v.get('selection_gate_passed'): raise PermissionError('V1 selection gate did not pass')
        expected=v['locked_sha256']; actual=hashlib.sha256(json.dumps(v['locked_settings'],sort_keys=True).encode()).hexdigest()
        if actual!=expected: raise ValueError('Frozen physics lock mismatch; create a separate explicit ablation')
        return dict(v['locked_settings'],name=name,source_candidate=v['source_candidate'])
    if name not in r['versions']: raise KeyError(name)
    return dict(r['common'],**r['versions'][name],name=name)


def pair_transform(name):
    settings=version(name)
    def transform(root,cfg):
        _forearm_transform(with_actuator=True)(root,cfg)
        if name==LEGACY: return
        collisions,_=_name_runtime_collision_geoms(root,cfg)
        contact=root.find('contact')
        if contact is None: contact=ET.SubElement(root,'contact')
        for surface,names in collisions.items():
            for geom in names:
                ET.SubElement(contact,'pair',name=f'contact_physics_{surface}_{geom}',
                              geom1=cfg.object['name']+'_geom',geom2=geom,
                              condim=str(settings['condim']),friction=' '.join(map(str,settings['pair_friction'])),
                              solref=' '.join(map(str,settings['solref'])),solimp=' '.join(map(str,settings['solimp'])),
                              margin=str(settings['margin']),gap=str(settings['gap']))
    return transform


def build_hand(name,*,diagnostic_timestep=None):
    cfg=phase3c07_scene_config(); settings=version(name)
    if diagnostic_timestep is not None and name=='PHYSICS_V1_NEAR_RIGID' and diagnostic_timestep!=settings['timestep']:
        raise ValueError('Frozen production timestep cannot be changed; use an explicit diagnostic version')
    cfg=replace(cfg,raw={**cfg.raw,'timestep':settings['timestep'] if diagnostic_timestep is None else diagnostic_timestep})
    scene=build_shadow_scene(cfg,model_transform=pair_transform(name))
    return scene


def assert_locked_model(scene,name,*,diagnostic=False):
    s=version(name); m=scene.model
    if not diagnostic and m.opt.timestep!=s['timestep']: raise ValueError('Production timestep changed')
    if m.opt.solver!=mujoco.mjtSolver.mjSOL_NEWTON or m.opt.integrator!=mujoco.mjtIntegrator.mjINT_EULER:
        raise ValueError('Solver/integrator lock violation')
    if m.opt.iterations!=s['iterations'] or m.opt.tolerance!=s['tolerance'] or m.opt.impratio!=s['impratio'] or m.opt.cone!=mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError('Numerical physics lock violation')
    if name!=LEGACY:
        expected=sum(len(x) for x in scene.collision_geoms.values())
        if m.npair!=expected: raise ValueError('Missing hand-object pair overrides')
        for key,expected in [('solref',s['solref']),('solimp',s['solimp']),('friction',s['pair_friction'])]:
            if not np.all(getattr(m,'pair_'+key)==np.asarray(expected)): raise ValueError('Contact physics lock violation: '+key)
        if not np.all(m.pair_dim==6): raise ValueError('Effective contact dimension changed')
        if not np.all(m.pair_margin==s['margin']) or not np.all(m.pair_gap==s['gap']):
            raise ValueError('Contact margin/gap changed')
        oid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,scene.config.object['name']+'_geom')
        if not np.all(m.geom_friction[oid]==s['sphere_friction']): raise ValueError('Sphere friction changed')
        for names in scene.collision_geoms.values():
            for geom in names:
                gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,geom)
                if not np.all(m.geom_friction[gid]==s['hand_friction']): raise ValueError('Hand friction changed')
    return True


def freeze_version(selection, path=None):
    """Never create a production alias from a failed/incomplete comparison."""
    if not selection.get('selection_gate_passed') or selection.get('selected_candidate') not in (IMP99,TC10):
        raise PermissionError('A passed physics selection gate is required')
    if not selection.get('all_hand_trials_complete') or not selection.get('simultaneous_mrl_validated'):
        raise PermissionError('Complete actual-hand multi-contact evidence is required')
    path=path or ROOT/'configs/PHYSICS_V1_NEAR_RIGID.yaml'
    if path.exists(): raise FileExistsError('Production physics is immutable; create a separately named ablation')
    selected=selection['selected_candidate']; settings=version(selected)
    settings.pop('name'); settings.pop('historical',None)
    lock=dict(name='PHYSICS_V1_NEAR_RIGID',source_candidate=selected,selection_gate_passed=True,
              locked_settings=settings,locked_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),
              policy='No task-driven physics tuning. Future compliance is a separate explicit versioned ablation.',
              selection_evidence=selection)
    # Called only after explicit evidence selection, never by the primary runner.
    path.write_text(yaml.safe_dump(lock,sort_keys=False),encoding='utf-8')
    return lock


def require_production_alias(name):
    if name!='PHYSICS_V1_NEAR_RIGID':
        raise PermissionError('Future production work requires the frozen V1 alias; candidates are diagnostics only')
    return version(name)
