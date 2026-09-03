"""Reset/state and frozen P0.5R contracts, plus saved full-exposure diagnostics."""
from copy import deepcopy
import hashlib
import mujoco
import numpy as np
import pytest

from seqgrasp import phase3cp05r as p, dynamic_reset as r, contact_physics as c


@pytest.fixture(scope='module')
def summary():
    if not (p.OUTPUT/'summary.json').exists(): pytest.skip('P0.5R local artifacts unavailable')
    return p.read('summary.json')


@pytest.fixture(scope='module')
def snap(summary): return summary['equilibrium_audit'][0]['snapshot']


def test_old_startup_acceleration_reproduces(summary):
    for a in summary['equilibrium_audit']:
        assert a['nominal']['max_acceleration']==pytest.approx(81.61685063369048,abs=1e-8)
        assert a['nominal']['max_speed']==0
        assert np.max(np.abs(a['nominal']['actuator_force']))<1e-12


def test_complete_dynamic_state_representation(snap):
    assert {'qpos_eq','qvel_eq','ctrl_eq','act_eq','qacc_eq','tendon_state','integration_state','state_sha256','model_sha256','cache_key','history'}<=set(snap)
    assert snap['state_sha256']==r.digest(snap['integration_state'])


def test_natural_utility_confirmation():
    m=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.002" gravity="0 0 0"/><worldbody><body><joint name="j" type="hinge" damping="1"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><actuator><position joint="j" kp="1"/></actuator></mujoco>')
    d=mujoco.MjData(m); d.qpos[0]=.2; d.ctrl[0]=.2; mujoco.mj_forward(m,d)
    g=r.DiagnosticGates('ENGINEERING_DIAGNOSTIC_ONLY',.001,.5,.02)
    out=r.settle_hand_to_dynamic_equilibrium(m,d,r.state_vector(m,d),d.ctrl.copy(),'TEST_ONLY',.1,g,hand_dofs=[0])
    assert out['snapshot']['converged'] and out['snapshot']['elapsed_s']>=.02-1e-10
    assert out['snapshot']['ctrl_eq']==[.2]


def test_no_ctrl_equals_settled_assumption(summary):
    for a in summary['equilibrium_audit']:
        assert np.max(abs(np.array(a['snapshot']['final_diagnostic']['ctrl_error'])))>.01
        assert a['ctrl_recenter_counterfactual']['max_acceleration']>.5
        assert a['counterfactual_dynamics_steps']==0


def test_natural_duration_and_unchanged_controller(snap):
    assert snap['converged'] and snap['confirmation_s']==.5
    assert snap['total_natural_duration_s']==pytest.approx(77.784)
    assert not snap['temporary_damping_used'] and not snap['velocity_overwrite_used']
    assert snap['original_damping_preserved']
    assert snap['ctrl_eq']==p.old.read(p.old.OUTPUT/'geometry.json')['initial_ctrl']


def test_original_damping_confirmation(snap):
    rows=p.old.load_trace(snap['restore_confirmation_trace'])
    assert len(rows)==251
    assert all(r.passes(x,p.gates()) for x in rows)


def test_complete_restore_preserves_qpos_qvel_ctrl_act(snap):
    s=p.old.setup_hand(c.IMP99,.002)
    with p.no_object_contact(s):
        r.restore_equilibrium(s.model,s.data,snap)
        for name,key in [('qpos','qpos_eq'),('qvel','qvel_eq'),('ctrl','ctrl_eq'),('act','act_eq')]:
            assert np.array_equal(getattr(s.data,name),snap[key])
        assert np.allclose(s.data.ten_length,snap['tendon_state']['length'])


def test_cache_hash_rejects_tampering(snap):
    broken=deepcopy(snap); broken['integration_state'][1]+=.001; s=p.old.setup_hand(c.IMP99,.002)
    with p.no_object_contact(s),pytest.raises(ValueError): r.restore_equilibrium(s.model,s.data,broken)


def test_cache_rejects_model_change(snap):
    s=p.old.setup_hand(c.IMP99,.002); s.model.actuator_gainprm[0,0]*=1.01
    with p.no_object_contact(s),pytest.raises(ValueError): r.restore_equilibrium(s.model,s.data,snap)


@pytest.mark.parametrize('key,value',[('max_speed',.002),('max_acceleration',1.),('max_acceleration',float('nan'))])
def test_engineering_guard_rejects_invalid(key,value):
    row=dict(max_speed=0,max_acceleration=0); row[key]=value
    assert not r.passes(row,p.gates())


def test_guard_requires_labeled_positive_gates():
    with pytest.raises(ValueError): r.DiagnosticGates('SCIENTIFIC',.001,.5,.5)
    with pytest.raises(ValueError): r.DiagnosticGates('ENGINEERING_DIAGNOSTIC_ONLY',0,.5,.5)
    with pytest.raises(ValueError): r.DiagnosticGates('ENGINEERING_DIAGNOSTIC_ONLY',float('nan'),.5,.5)


def test_no_object_mechanism_restores_parameters():
    s=p.old.setup_hand(c.IMP99,.002); before=r.model_signature(s.model)
    with p.no_object_contact(s):
        assert np.all(s.model.pair_margin==-1) and np.all(s.model.pair_gap==1)
        assert all(not x['active_constraint'] for x in r.raw_diagnostic(s.model,s.data,p.hand_dofs(s))['contacts'])
    assert r.model_signature(s.model)==before


def test_settled_fk_is_recomputed(summary):
    for a in summary['equilibrium_audit']:
        assert len(a['fk_displacements'])==13
        assert max(x['position_displacement_m'] for x in a['fk_displacements'])>.001
        assert max(x['orientation_displacement_deg'] for x in a['fk_displacements'])>1


def test_new_positive_geometry_not_old_40um(summary):
    g=summary['geometry']; assert g['gap_m']==.0004
    assert min(g['all_hand_gaps_m'])>=.0004-1e-8
    assert g['sphere_center_palm_m']!=p.old.read(p.old.OUTPUT/'geometry.json')['center_palm_m']
    assert np.allclose(g['qvel'][:-6],summary['equilibrium_audit'][0]['snapshot']['qvel_eq'][:-6])


def test_zero_command_prehold(summary):
    for t in summary['trials']:
        assert t['prehold']['contact_absent']
        rows=p.old.load_trace(t['trace']); initial=np.array(rows[0]['ctrl'])
        assert all(np.array_equal(row['ctrl'],initial) for row in rows if row['time_s']<=.252+1e-10)


def test_common_equilibrium_and_trial_state_hashes(summary):
    assert len({a['snapshot']['state_sha256'] for a in summary['equilibrium_audit']})==1
    assert len({x['initial_state_sha256'] for x in summary['trials']})==1
    assert len({x['command_sha256'] for x in summary['trials']})==1


@pytest.mark.parametrize('name,ref',[(c.IMP99,[.02,1]),(c.TC10,[.01,1])])
def test_frozen_contact_settings(name,ref):
    assert c.version(name)['solref']==ref
    s=c.build_hand(name); assert c.assert_locked_model(s,name)
    assert np.all(s.model.pair_solimp==[.99,.99,.001,.5,2])
    assert np.all(s.model.pair_dim==6)
    assert np.all(s.model.pair_friction==[.5,.5,.01,.003,.003])


def test_all_three_timesteps_requested(summary):
    assert len(summary['trials'])==6
    assert set((x['physics_name'],x['nominal_dt_s']) for x in summary['trials'])=={(n,dt) for n in p.config()['candidates'] for dt in [.001,.002,.004]}


def test_equal_duration_grid_no_variable_timestep():
    cc=p.control_schedule(); assert cc['prehold_s']==.252
    for dt in [.001,.002,.004]:
        for name in ['prehold_s','ramp_s','hold_s']: assert abs(cc[name]/dt-round(cc[name]/dt))<1e-9


def test_full_loaded_exposure_at_1_and_2ms(summary):
    for t in summary['trials']:
        if t['nominal_dt_s']!=.004:
            assert t['completed'] and t['elapsed_s']==pytest.approx(3.252)
            assert t['loaded_integration_steps']>100
            assert t['sustained_loaded_hold'] and t['steady'] is not None
            assert not t['simultaneous_mrl_tail']


def test_no_steady_metrics_for_censored_trials(summary):
    for t in summary['trials']:
        if not t['completed']: assert t['steady'] is None and t['loaded_integration_steps']==0


def test_raw_qacc_tendon_force_and_wrench_logging(summary):
    t=next(x for x in summary['trials'] if x['nominal_dt_s']==.002)
    rows=p.old.load_trace(t['trace'])
    assert {'qacc','act','tendon_actuator_force','ctrl','actuator_saturation_fraction','weld_force_world_n','weld_torque_world_nm','solver'}<=set(rows[-1])
    assert all(row['welded'] for row in rows)
    assert all(c['dim']==6 and len(c['local_wrench_on_geom2'])==6 for row in rows for c in row['contacts'])


def test_force_balance_residual(snap):
    assert np.max(np.abs(snap['final_diagnostic']['force_balance_residual']))<1e-10


def test_production_reset_repaired_but_4ms_guard_not_hidden(summary):
    assert summary['selection']['reset_solved_at_production_dt']
    assert not summary['selection']['cross_timestep_reset_valid']
    for t in summary['trials']:
        if t['nominal_dt_s']==.004:
            assert t['stop_reason']=='INVALID_PREHOLD_DYNAMICS' and t['prehold']['max_speed']>.001


def test_failed_selection_cannot_create_v1(summary,tmp_path):
    with pytest.raises(PermissionError): c.freeze_version(summary['selection'],tmp_path/'V1.yaml')
    assert not (p.ROOT/'configs/PHYSICS_V1_NEAR_RIGID.yaml').exists()


def test_lock_policy_candidates_not_production():
    with pytest.raises(PermissionError): c.require_production_alias(c.IMP99)


def test_geometric_resource_results_preserved(summary):
    assert summary['resource_fractions']==dict(thumb=.9559782183972225,index=1.,opposition=.9665998246424643)
    assert len(summary['static_results_preserved'])==5


@pytest.mark.parametrize('key',['receiver','b03','weld_release','handoff','object_B','rl','shape','skin','resource_recompute','storage_search'])
def test_no_out_of_scope_experiments(key):
    assert p.read('protocol.json')['scope'][key] is False


def test_p05_outputs_preserved_exactly(summary):
    assert summary['p05_outputs_preserved']
    for path,sha in p.read('protocol.json')['preserved_p05_outputs'].items():
        assert hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==sha


def test_historical_regressions_only_planned(summary):
    assert summary['historical_regression_plan']['executed'] is False
    assert summary['selection']['historical_dynamic_regressions_executed'] is False
