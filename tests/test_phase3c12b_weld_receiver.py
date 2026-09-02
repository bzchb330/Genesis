"""Measured actuation and weld/contact-wrench contracts; no alternate receiver."""
import inspect
import json
import subprocess

import mujoco
import numpy as np
import pytest

from seqgrasp import phase3c12b as b


def remote_sphere():
    s=b.build(); b.set_object_pose(s,[.8,0,.8]); b.set_fixture(s,True)
    s.data.ctrl[:]=b.a.actuator_target_from_qpos(s,s.data.qpos)
    mujoco.mj_forward(s.model,s.data); return s


def test_compiled_actuator_type_extraction():
    d=b.read('actuation_audit.json'); assert len(d['actuators'])==21
    assert all(r['type'].startswith('position servo') for r in d['actuators'])


def test_ctrl_semantics_command_response_force():
    s=remote_sphere(); i=mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_ACTUATOR,'rh_A_MFJ3')
    j=s.model.actuator_trnid[i,0]; adr=s.model.jnt_qposadr[j]; before=s.data.qpos[adr]
    s.data.ctrl[i]=s.data.actuator_length[i]+.03; mujoco.mj_forward(s.model,s.data)
    assert s.data.actuator_force[i]==pytest.approx(.03)
    for _ in range(30): mujoco.mj_step(s.model,s.data)
    mujoco.mj_forward(s.model,s.data)
    assert s.data.qpos[adr]>before
    assert s.data.actuator_force[i]==pytest.approx(s.model.actuator_gainprm[i,0]*(s.data.ctrl[i]-s.data.actuator_length[i]))


def test_transmission_tendon_mapping():
    s=remote_sphere(); matrix=b.transmission_matrix(s)
    s.data.qvel[:]=np.linspace(-.1,.1,s.model.nv); mujoco.mj_forward(s.model,s.data)
    np.testing.assert_allclose(matrix@s.data.qvel,s.data.actuator_velocity,atol=1e-12)


def test_coupled_joint_audit():
    d=b.read('actuation_audit.json'); assert len(d['tendons'])==4
    for tendon in d['tendons']: assert [r['coefficient'] for r in tendon['terms']]==[1.,1.]
    assert d['transmission_rank']==21 and d['hand_dofs']==25


def test_forcerange_extraction():
    d=b.read('actuation_audit.json'); m=next(r for r in d['actuators'] if r['name']=='rh_A_MFJ0')
    assert m['forcerange']==[-1,1] and m['ctrlrange']==[0,3.1415]


def test_actuator_saturation_not_ctrl_clipping():
    s=remote_sphere(); s.data.ctrl[:]=s.model.actuator_ctrlrange[:,1]; mujoco.mj_forward(s.model,s.data)
    ratio,active=b.saturation(s)
    np.testing.assert_equal(active,s.model.actuator_forcelimited.astype(bool)&(ratio>=1-1e-8))
    assert np.any(~active)  # A bound-valued command is not sufficient evidence.


def test_fixed_sphere_constraint():
    s=remote_sphere(); anchor=s.data.mocap_pos.copy()
    for _ in range(100): mujoco.mj_step(s.model,s.data)
    assert s.data.eq_active[s.fixture_eq_id]
    np.testing.assert_equal(s.data.mocap_pos,anchor)
    assert np.linalg.norm(s.data.xpos[s.object_body_id]-anchor[0])<b.config().temporary_support['fixed_pose_numerical_tolerance_m']


def test_virtual_command_offset():
    cfg=b.config(); assert cfg.primitive['virtual_offsets']==[0,.01,.025,.05,.1,.2,.3,.4]
    assert cfg.primitive['total_steps']==500
    assert 'normal_virtual_direction' in inspect.getsource(b.tangent_setup)


def test_persistent_actuator_error():
    s=remote_sphere(); i=mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_ACTUATOR,'rh_A_MFJ0')
    s.data.ctrl[i]=s.data.actuator_length[i]+.1; mujoco.mj_forward(s.model,s.data)
    assert s.data.ctrl[i]-s.data.actuator_length[i]==pytest.approx(.1)
    assert s.data.actuator_force[i]==pytest.approx(.05)


def test_persistent_contact_force_truthful_descriptor():
    d=b.read('fixed_sphere_primitives.json')
    for r in d['rows']:
        assert r['persistent_final_100']==(r['minimum_tail_force_n']>1e-9)
        assert not r['admissible_primitive'] or (r['persistent_final_100'] and not r['saturated'] and not r['contact_switch'])


def test_contact_identity_tracking():
    rows=b.load_series((b.OUTPUT/'primitives/little_07.npz').relative_to(b.ROOT))
    assert len(rows)==500
    for r in rows:
        for c in r['contacts']: assert len(c['geom_ids'])==2 and len(c['inward_normal_world'])==3


def test_fixed_sphere_primitive_frozen():
    d=b.read('fixed_sphere_primitives.json'); assert d['total_trials']==32 and d['steps_each']==500
    assert all(r['physics_unchanged'] for r in d['rows'])
    assert b.read('frozen_protocol.json')['frozen_before_primitive_outcomes']


def test_simultaneous_mrl_ramp():
    assert b.smooth_ramp(0,200)==0 and b.smooth_ramp(100,200)==.5 and b.smooth_ramp(200,200)==1
    src=inspect.getsource(b.run_construction); assert 'initial+smooth_ramp' in src
    assert b.read('receiver_protocol.json')['ramp_steps']==200


def test_temporary_weld_creation():
    s=remote_sphere(); assert s.model.eq_type[s.fixture_eq_id]==mujoco.mjtEq.mjEQ_WELD
    assert s.model.body_mocapid[s.model.eq_obj2id[s.fixture_eq_id]]==s.fixture_mocap_id


def test_deterministic_weld_release_only_constraint_changes():
    s=remote_sphere(); q=s.data.qpos.copy(); v=s.data.qvel.copy(); ctrl=s.data.ctrl.copy()
    b.set_fixture(s,False)
    np.testing.assert_equal(s.data.qpos,q); np.testing.assert_equal(s.data.qvel,v); np.testing.assert_equal(s.data.ctrl,ctrl)
    assert not s.data.eq_active[s.fixture_eq_id]


def test_weld_force_logging():
    s=remote_sphere()
    for _ in range(100): mujoco.mj_step(s.model,s.data)
    mujoco.mj_forward(s.model,s.data); f,t=b.weld_wrench(s)
    assert f[2]>0 and np.isfinite(t).all()
    b.set_fixture(s,False); mujoco.mj_forward(s.model,s.data)
    np.testing.assert_allclose(b.weld_wrench(s),0,atol=1e-12)


def test_contact_wrench_aggregation():
    s,source=b.receiver_setup(); contacts,f,t,_=b.contact_wrenches(s)
    assert contacts
    np.testing.assert_allclose(f,np.sum([r['force_on_sphere_world_n'] for r in contacts],axis=0),atol=1e-12)
    np.testing.assert_allclose(t,np.sum([r['torque_about_sphere_com_nm'] for r in contacts],axis=0),atol=1e-12)


def test_gravitational_wrench():
    s=remote_sphere(); row=b.record(s,0,'TEST',s.data.xpos[s.object_body_id].copy())
    np.testing.assert_allclose(row['gravity_force_world_n'],[0,0,-.08025787482217676])


def test_counterfactual_free_net_force_excludes_weld():
    s=remote_sphere(); row=b.record(s,0,'TEST',s.data.xpos[s.object_body_id].copy())
    np.testing.assert_allclose(row['free_net_force_world_n'],np.array(row['hand_force_world_n'])+row['gravity_force_world_n'])


def test_counterfactual_free_net_torque_excludes_weld():
    s=remote_sphere(); row=b.record(s,0,'TEST',s.data.xpos[s.object_body_id].copy())
    np.testing.assert_equal(row['free_net_torque_world_nm'],row['hand_torque_world_nm'])


def test_sphere_linear_angular_acceleration():
    s=remote_sphere(); row=b.record(s,0,'TEST',s.data.xpos[s.object_body_id].copy())
    np.testing.assert_allclose(row['counterfactual_linear_acceleration_mps2'],[0,0,-9.81])
    np.testing.assert_allclose(row['counterfactual_angular_acceleration_radps2'],0,atol=1e-10)


def test_post_release_support_requires_no_weld():
    row=dict(welded=True,topology=['middle','little'],contacts=[],maximum_penetration_m=0)
    assert not b.retained(row); row['welded']=False; assert b.retained(row)
    row['topology']=[]; assert not b.retained(row)


def test_failure_taxonomy():
    assert len(b.FAILURES)==10 and 'STABLE_RECEIVER' in b.FAILURES
    result=b.read('release_results.json')
    if not result['executed']: assert all(x is None for x in result['checkpoints'].values())


def test_modeled_realized_contact_comparison():
    s,source=b.receiver_setup(); initial=b.a.network_from_scene(s,'INITIAL',b.SURFACES)
    for con in source['network']['contacts']:
        r=next(x for x in initial['contacts'] if x['geom_ids']==con['geom_ids'])
        np.testing.assert_allclose(r['inward_normal_palm'],con['inward_normal_palm'],atol=1e-10)


def test_no_shape_retest(): assert not b.contract()['shape_retest']
def test_no_skin(): assert not b.contract()['skin']
def test_no_rl(): assert not b.contract()['rl']
def test_no_handoff(): assert not b.contract()['handoff']
def test_no_object_b(): assert not b.contract()['object_B']


def test_phase3c12a_backward_compatibility():
    diff=subprocess.check_output(['git','diff','6f550dede4b94b3e755bb3b1b208a1880a21a562','--','seqgrasp/phase3c12a.py','configs/phase3C12A_contact_gravity_wrench.yaml','docs/figures/phase3C12A'],cwd=b.ROOT,text=True)
    assert diff==''
