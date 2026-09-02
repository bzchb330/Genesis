"""Phase 3C-1.2A equations, measured-artifact integrity, and scope guards."""
import inspect
import json
import subprocess

import mujoco
import numpy as np
import pytest

from seqgrasp import phase3c12a as c


def read(name):
    return json.loads((c.OUTPUT/name).read_text())


def network(normals, arms=None):
    if arms is None: arms=-np.asarray(normals)*.0125
    return dict(weight_n=1.,contacts=[dict(surface=str(i),inward_normal_palm=n,
        lever_arm_palm_m=r,friction=.5,runtime_friction=[.5,.5,.01,.003,.003])
        for i,(n,r) in enumerate(zip(normals,arms))])


def test_previous_calibration_reconstruction():
    d=read('calibration_autopsy.json')
    assert len(d['rows'])==66 and d['maximum_force_reconstruction_error_n']<1e-12
    assert all(r['commanded_joint_displacement_rad']==0 for r in d['rows'])


def test_geom_pair_identity_tracking():
    assert c.pair_changed([{'geom_ids':[1,2]}],1,2) is False
    assert c.pair_changed([{'geom_ids':[1,3]}],1,2) is True
    assert c.pair_changed([],1,2) is False  # disappearance != switching


def test_contact_position_tracking():
    rows=read('calibration_autopsy.json')['rows']
    assert any(r['contact_tangential_migration_m'] and r['contact_tangential_migration_m']>0 for r in rows)
    assert all(len(x['position_world_m'])==3 for r in rows for x in r['contacts'])


def test_contact_normal_tracking():
    for row in read('calibration_autopsy.json')['rows']:
        for x in row['contacts']:
            assert np.isclose(np.linalg.norm(x['inward_normal_world']),1)
    source=inspect.getsource(c.contact_rows)
    assert '1 if g2 == oid else -1' in source


def test_cartesian_local_normal_approach():
    rows=read('corrected_calibration.json')['rows']
    successful=[r for r in rows if r['command_offset_mm']>0 and r['ik_error_m']<1e-8]
    assert successful
    assert all(r['commanded_motion']['normal_motion_fraction']>.999 for r in successful)


def test_ik_jacobian_validation():
    scene=c.build_forearm_scene(with_actuator=True).scene
    mujoco.mj_forward(scene.model,scene.data)
    joint=scene.joint_ids['index'][-1]; body=int(scene.model.jnt_bodyid[joint])
    local=np.array([.003,.004,.005]); point=c.material_point(scene,body,local)
    jac=np.zeros((3,scene.model.nv)); mujoco.mj_jac(scene.model,scene.data,jac,None,point,body)
    address=scene.model.jnt_qposadr[joint]; dof=scene.model.jnt_dofadr[joint]
    scene.data.qpos[address]+=1e-7; mujoco.mj_forward(scene.model,scene.data)
    np.testing.assert_allclose((c.material_point(scene,body,local)-point)/1e-7,jac[:,dof],atol=1e-8)


def test_normal_motion_fraction():
    d=c.motion_components([3,4,0],[1,0,0]); assert d['normal_motion_fraction']==.6 and d['tangential_drift_m']==4
    assert c.motion_components([0,0,0],[1,0,0])['normal_motion_fraction'] is None


def test_command_offset_calibration():
    d=read('corrected_calibration.json')
    assert d['command_offsets_mm']==[0,.025,.05,.075,.1,.15,.2,.3,.4,.5]
    assert len(d['rows'])==60
    for r in d['rows']:
        assert len(r['timeline'])<=50
        assert r['branch_terminated']==r['contact_pair_switched']
        assert np.isfinite(r['object_acceleration_mps2']).all()


def test_object_weight():
    scene=c.build_forearm_scene(with_actuator=True).scene
    assert c.object_weight(scene)==pytest.approx(scene.model.body_mass[scene.object_body_id]*9.81)
    assert c.object_weight(scene)==pytest.approx(.08025787482217676)


def test_normal_cone_construction():
    d=c.normal_cone([[1,0,0],[0,1,0]],[2,3,-4])
    np.testing.assert_allclose(d['closest_vector_n'],[2,3,0]); assert d['residual_n']==4


def test_cone_membership():
    assert c.normal_cone([[0,0,1]],[0,0,2])['feasible']
    assert not c.normal_cone([[0,0,1]],[0,0,-2])['feasible']


def test_angular_distance_to_convex_cone():
    assert c.normal_cone([[1,0,0]],[-1,0,0])['angular_distance_deg']==180
    assert c.normal_cone([[1,0,0]],[0,1,0])['angular_distance_deg']==90
    assert c.normal_cone([[1,0,0]],[1,1,0])['angular_distance_deg']==pytest.approx(45)
    assert c.normal_cone([[1,0,0]],[0,0,0])['angular_distance_deg']==0


def test_frictionless_equilibrium_full_torque():
    good=c.equilibrium(network([[0,0,1]]),[0,0,-1],0)
    assert good['feasible']; np.testing.assert_allclose(good['torque_residual_nm'],0,atol=1e-12)
    assert not c.equilibrium(network([[0,0,1]],[[.01,0,0]]),[0,0,-1],0)['feasible']


def test_friction_scaled_equilibrium_sweep():
    d=c.friction_curve(network([[1,0,0]],[[0,0,0]]),[-1/np.sqrt(1.04),0,-.2/np.sqrt(1.04)])
    assert [r['scale'] for r in d['rows']]==[0,.1,.25,.5,.75,1]
    assert d['minimum_tested_feasible_scale']==.5


def test_friction_utilization_and_compiled_wrench():
    assert c.friction_utilization(2,[.3,.4],.5)==.5
    assert c.friction_utilization(0,[0,0],.5) is None
    sol=c.compiled_wrench_equilibrium(network([[0,0,1]]),[0,0,-1])
    assert sol['feasible'] and sol['rho_max']<1e-6
    assert np.linalg.norm(sol['torque_residual_nm'])<1e-9


def test_reachable_gravity_orientation_generation():
    scene=c.build_forearm_scene(with_actuator=True).scene; bounds=c.orientation_bounds(scene)
    points=c.reachable_orientations(scene); assert points.shape==(45,3)
    assert np.all(points>=bounds[:,0]) and np.all(points<=bounds[:,1])
    np.testing.assert_equal(scene.model.opt.gravity,[0,0,-9.81])


def test_storage_orientation_search():
    for a in read('mechanics_audits.json')['candidates']:
        assert a['sample_count']==48 and len(a['rows'])==48
        assert a['best']['frictionless']['feasible']
        assert a['oriented_geometry_check']['all_pairs_preserved']
        assert all(e['position_error_m']<1e-10 for e in a['oriented_geometry_check']['errors'])


def test_transport_storage_orientation_comparison():
    for a in read('mechanics_audits.json')['candidates']:
        t=a['transport_comparison']; assert t['source_state_id']=='C07_STATE_00000'
        angle=np.rad2deg(np.arccos(np.clip(np.dot(t['transport_gravity_direction'],a['best']['gravity_direction_palm']),-1,1)))
        assert t['gravity_direction_difference_deg']==pytest.approx(angle)


def test_old_role_t_implementation_audit():
    a=read('old_role_t_audit.json')
    assert not a['thumb_required_by_prefilter'] and not a['thumb_required_by_closure']
    assert all(r['thumb_force_n']==0 for r in a['candidates'])


def test_role_t_true_requires_thumb():
    n={'contacts':[{'surface':'ring','normal_force_n':1},{'surface':'little','normal_force_n':1}]}
    assert not c.is_true_role_t(n)
    n['contacts'][0]['surface']='thumb'; assert c.is_true_role_t(n)
    n['contacts'][0]['normal_force_n']=0; assert not c.is_true_role_t(n)


def test_role_t_true_preserves_index_middle():
    a=read('true_role_t_search.json'); assert a['evaluated']==18
    for row in a['rows']:
        if row['true_role_preloaded']:
            assert not {'index','middle'}&{x['surface'] for x in row['network']['contacts']}
    w=read('true_role_t_workspace.json')
    assert 'thumb_point_palm_m' not in w['role_t_true']['joint_acquisition']['representative_pairs'][0]


def test_role_mrl_role_t_comparison():
    a=read('mechanics_audits.json')['candidates']; m,t=a[0],a[3]
    assert m['network']['candidate_id']=='ROLE_MRL_05'
    assert t['network']['candidate_id']=='ROLE_T_TRUE_07'
    assert m['best']['actual_friction']['rho_max']==t['best']['actual_friction']['rho_max']==0
    assert m['normal_cone_generator_span_deg']>t['normal_cone_generator_span_deg']


def test_archived_mrl_exact_workspace():
    w=c.read_old('resource_workspace_audit.json')['retained_fraction']
    assert w==dict(thumb=.9559782183972225,index=1.0,opposition=.9665998246424643)


def test_no_object_b():
    assert not c.phase_contract()['object_B']
    scene=c.build_forearm_scene(with_actuator=True).scene
    assert not any('object_b' in (mujoco.mj_id2name(scene.model,mujoco.mjtObj.mjOBJ_BODY,i) or '').lower() for i in range(scene.model.nbody))


def test_no_handoff():
    assert not c.phase_contract()['handoff'] and not c.phase_contract()['receiver_hold_dynamics']
    assert inspect.getsource(c).count('mujoco.mj_step(')==1


def test_no_rl():
    assert not c.phase_contract()['rl'] and 'stable_baselines' not in inspect.getsource(c)


def test_no_trajectory_optimizer():
    assert not c.phase_contract()['trajectory_optimizer'] and not c.phase_contract()['contact_implicit_optimizer']
    assert c.phase_contract()['offline_equilibrium_and_orientation_optimization']


def test_no_skin():
    assert not c.phase_contract()['skin']


def test_no_contact_physics_changes():
    scene=c.build_forearm_scene(with_actuator=True).scene
    assert scene.model.opt.timestep==.002 and scene.model.opt.cone==1
    gid=c._object_geom_id(scene); np.testing.assert_equal(scene.model.geom_friction[gid],[.5,.01,.003])
    for field in ('geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_condim'):
        assert not any(field in line and '=' in line and not '==' in line for line in inspect.getsource(c).splitlines())


def test_no_shape_dynamics():
    assert not c.phase_contract()['shape_dynamics'] and 'build_shape_scene' not in inspect.getsource(c)


def test_phase3c11_backward_compatibility():
    result=subprocess.run(['git','diff','6ca46034a743ba265b7ef58be452decdcb138f33','--','seqgrasp/phase3c11.py','configs/phase3C11_preload_shape_resource.yaml'],cwd=c.ROOT,capture_output=True,text=True,check=True)
    assert result.stdout==''
    assert c.read_old('preloaded_B03_results.json')['classification']=='PR-E'
