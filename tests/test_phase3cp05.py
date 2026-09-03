"""P0.5 protocol/physics/logging regression; no new comparison simulations."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from seqgrasp import contact_physics as c, phase3cp05 as p


@pytest.fixture(scope='module')
def models():
    return {n:c.build_hand(n) for n in (c.LEGACY,c.IMP99,c.TC10)}


@pytest.fixture(scope='module')
def summary():
    if not (p.OUTPUT/'summary.json').exists(): pytest.skip('Local P0.5 artifacts not present')
    return p.read(p.OUTPUT/'summary.json')


def test_registry_exact_three_versions():
    assert set(c.registry()['versions'])=={c.LEGACY,c.IMP99,c.TC10}
    assert c.registry()['production_alias'] is None


def test_legacy_preservation(models):
    s=p.native.a.build_forearm_scene(with_actuator=True).scene
    assert p.native.physics_fingerprint(models[c.LEGACY])==p.native.physics_fingerprint(s)
    assert models[c.LEGACY].model.npair==0


@pytest.mark.parametrize('name,tc,imp',[(c.LEGACY,.02,[.9,.95,.001,.5,2]),(c.IMP99,.02,[.99,.99,.001,.5,2]),(c.TC10,.01,[.99,.99,.001,.5,2])])
def test_exact_candidate_parameters(name,tc,imp):
    assert c.version(name)['solref']==[tc,1]
    assert c.version(name)['solimp']==imp


@pytest.mark.parametrize('name',[c.IMP99,c.TC10])
def test_explicit_override_all_surfaces(models,name):
    s=models[name]; m=s.model
    assert set(s.collision_geoms)=={'thumb','index','middle','ring','little','palm'}
    assert m.npair==31
    oid=p.native.a._object_geom_id(s)
    assert np.all((m.pair_geom1==oid)|(m.pair_geom2==oid))
    expected={g for x in s.collision_geoms for g in p.native.a._geom_ids(s,x)}
    actual={int(b if a==oid else a) for a,b in zip(m.pair_geom1,m.pair_geom2)}
    assert expected==actual


@pytest.mark.parametrize('name',[c.IMP99,c.TC10])
def test_friction_and_native_geometry_unchanged(models,name):
    assert p.native.physics_fingerprint(models[name])==p.native.physics_fingerprint(models[c.LEGACY])
    assert np.all(models[name].model.pair_friction==[.5,.5,.01,.003,.003])
    assert np.all(models[name].model.pair_dim==6)
    assert np.all(models[name].model.pair_solreffriction==0)


@pytest.mark.parametrize('name',[c.LEGACY,c.IMP99,c.TC10])
def test_production_numerical_settings(models,name):
    assert c.assert_locked_model(models[name],name)
    assert models[name].model.opt.timestep==.002


def test_p0_regression(summary):
    assert summary['regression']['passed']
    assert len(summary['regression']['rows'])==8
    for r in summary['regression']['rows']:
        assert abs(r['p0_difference_m'])<=p.config()['regression_absolute_tolerance_m']
        assert r['mean']['normal_force_n']==pytest.approx(r['load_n'],rel=1e-6)


def test_impact_protocol_frozen():
    assert p.config()['impact']['heights_m']==[.0005,.001,.002,.005]
    assert p.config()['impact']['duration_s']==1


def test_dynamic_logging(summary):
    assert len(summary['impact'])==24
    for x in summary['impact']:
        rows=p.load_trace(x['trace'])
        assert rows[0]['physics_name']==x['physics_name']
        assert len(rows)==round(1/x['dt_s'])+1
        assert {'normal_force_n','contact_wrench_world','kinetic_energy_j','potential_energy_j','contact_work_j','energy_residual_j','contacts'}<=set(rows[0])
        assert np.isfinite([r['normal_force_n'] for r in rows]).all()


def test_contact_chatter_counting():
    e=p.events([False,True,False,True,False],.001)
    assert e['makes']==2 and e['breaks']==2 and e['rapid_intervals']==3
    assert e['contact_durations_s']==[.001,.001]
    e=p.events([False,True],.001)
    assert e['last_episode_right_censored'] and e['contact_durations_s']==[0]


def test_energy_diagnostics(summary):
    for x in summary['impact']:
        assert x['energy_residual_j']==pytest.approx(x['final_energy_j']-x['initial_energy_j']-x['contact_work_j'])


def test_fixed_sphere_collision_free_start(summary):
    g=p.read(p.OUTPUT/'geometry.json'); assert g['initial_contacts']==[]
    assert min(g['all_hand_object_gaps_m'])>0
    for name in p.config()['candidates']:
        s=p.setup_hand(name,.002)
        assert s.data.eq_active[s.fixture_eq_id]
        assert s.data.ncon==0
        assert np.array_equal(s.data.qvel,np.zeros(s.model.nv))


def test_tendon_transmission(models):
    t=p.native.transmission_matrix(models[c.LEGACY])
    for name in (c.IMP99,c.TC10):
        assert np.array_equal(t,p.native.transmission_matrix(models[name]))
    m=models[c.LEGACY].model
    assert sum(m.actuator_trntype==mujoco.mjtTrn.mjTRN_TENDON)==4


def test_identical_controller_no_candidate_specific_tuning(summary):
    assert len({h['command_sha256'] for h in summary['hand']})==1
    assert p.config()['hand']['virtual_actuator_offset']==.015
    assert p.config()['hand']['ramp_s']==1 and p.config()['hand']['hold_s']==2


def test_full_hand_logging(summary):
    required={'qpos','qvel','ctrl','actuator_force','tendon_length','tendon_velocity','actuator_saturated','weld_force_world_n','weld_torque_world_nm','maximum_penetration_m','contacts'}
    for h in summary['hand']:
        rows=p.load_trace(h['trace']); assert required<=set(rows[0])
        assert all(r['welded'] for r in rows)
        for r in rows:
            for x in r['contacts']:
                assert {'position_world_m','inward_normal_world','distance_m','local_wrench_on_geom2','normal_force_n','tangential_force_n'}<=set(x)
                assert x['dim']==6 and x['friction']==[.5,.5,.01,.003,.003]


def test_exact_overlap_formula():
    r=.0125; delta=.005; g=p.overlap_geometry(delta,r)
    assert g['a_m']==pytest.approx(np.sqrt(2*r*delta-delta**2))
    assert g['a_m']!=pytest.approx(np.sqrt(2*r*delta))
    assert p.overlap_geometry(0)['a_m']==0


def test_force_and_topology_variances(summary):
    for h in summary['hand']:
        rows=p.load_trace(h['trace'])
        fn=[sum(c['normal_force_n'] for c in r['contacts']) for r in rows]
        assert h['total_force_variance_tail_n2']==pytest.approx(np.var(fn))
        assert h['contact_count_variance_all']>=0


def test_contact_migration():
    cs=[dict(geom_names=['a','b'],normal_force_n=1,position_world_m=[0,0,0],inward_normal_world=[1,0,0]),
        dict(geom_names=['a','b'],normal_force_n=1,position_world_m=[.001,0,0],inward_normal_world=[0,1,0])]
    x=p.contact_migration([dict(contacts=[c]) for c in cs])['a|b']
    assert x['maximum_displacement_from_first_m']==.001
    assert x['maximum_normal_angle_from_first_deg']==90


def test_equal_planned_physical_duration():
    assert p.config()['timesteps_s']==[.001,.002,.004]
    assert [round(3/t) for t in p.config()['timesteps_s']]==[3000,1500,750]
    assert all(round(3/t)*t==3 for t in p.config()['timesteps_s'])


def test_censored_runs_not_called_steady(summary):
    for h in summary['hand']:
        if not h['completed']:
            assert h['steady_penetration_m'] is None and h['steady_total_normal_force_n'] is None
            assert 'not steady' in h['window_semantics']


def test_failed_gate_cannot_create_v1(tmp_path,summary):
    dest=tmp_path/'v1.yaml'
    with pytest.raises(PermissionError): c.freeze_version(summary['selection'],dest)
    assert not dest.exists()


def test_incomplete_hand_cannot_create_v1(tmp_path):
    with pytest.raises(PermissionError):
        c.freeze_version(dict(selection_gate_passed=True,selected_candidate=c.IMP99,all_hand_trials_complete=False),tmp_path/'v1.yaml')


def test_hypothetical_valid_lock_and_tamper(tmp_path,monkeypatch):
    # Isolated test fixture only; not a production scientific selection.
    dest=tmp_path/'configs/PHYSICS_V1_NEAR_RIGID.yaml'; dest.parent.mkdir()
    sel=dict(selection_gate_passed=True,selected_candidate=c.IMP99,all_hand_trials_complete=True,simultaneous_mrl_validated=True)
    lock=c.freeze_version(sel,dest); monkeypatch.setattr(c,'ROOT',tmp_path)
    assert c.version('PHYSICS_V1_NEAR_RIGID')['source_candidate']==c.IMP99
    with pytest.raises(FileExistsError): c.freeze_version(sel,dest)
    with pytest.raises(ValueError): c.build_hand('PHYSICS_V1_NEAR_RIGID',diagnostic_timestep=.001)
    lock['locked_settings']['solref'][0]=.001; dest.write_text(yaml.safe_dump(lock))
    with pytest.raises(ValueError): c.version('PHYSICS_V1_NEAR_RIGID')


def test_production_requires_alias():
    with pytest.raises(PermissionError): c.require_production_alias(c.IMP99)


def test_model_lock_rejects_mutation():
    s=c.build_hand(c.IMP99); s.model.pair_friction[0,0]=.6
    with pytest.raises(ValueError): c.assert_locked_model(s,c.IMP99)
    s=c.build_hand(c.IMP99); s.model.opt.timestep=.001
    with pytest.raises(ValueError): c.assert_locked_model(s,c.IMP99)


@pytest.mark.parametrize('key',['receiver','b03','handoff','object_B','rl','shape','skin','weld_release'])
def test_out_of_scope_stays_gated(summary,key):
    assert p.read(p.OUTPUT/'frozen_protocol.json')['scope'][key] is False


def test_optional_tangential_gated(summary):
    assert not summary['selection']['optional_tangential_executed']
    assert not list(p.OUTPUT.glob('*/tangential_*.npz'))


def test_p0_all_artifacts_preserved(summary):
    f=p.read(p.OUTPUT/'frozen_protocol.json')
    for path,sha in f['preserved_p0_files'].items():
        assert hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==sha
    assert summary['p0_preserved']


def test_no_v1_for_current_failed_selection(summary):
    assert not (p.ROOT/'configs/PHYSICS_V1_NEAR_RIGID.yaml').exists()
    assert summary['selection']['selected_candidate'] is None


def test_raw_trace_identification(summary):
    for row in summary['hand']+summary['impact']+summary['regression']['rows']:
        with np.load(p.ROOT/row['trace'],allow_pickle=False) as data:
            assert str(data['physics_name'])==row['physics_name']


def test_figure_manifest(summary):
    f=p.read(p.OUTPUT/'figures.json')
    assert f['count']==20 and f['physics_steps']==0
    for path in f['figures']:
        assert (p.ROOT/path).read_bytes().startswith(b'%PDF')


def test_video_manifest(summary):
    v=p.read(p.OUTPUT/'videos.json')
    assert len(v['generated'])==2 and v['physics_steps']==0
    assert not v['hand_videos_generated']


def test_primary_and_summary_agree_on_censoring(summary):
    primary=p.read(p.OUTPUT/'primary_results.json')
    assert primary['hand']==summary['hand']


def test_startup_audit_is_not_new_dynamics(summary):
    a=p.read(p.OUTPUT/'startup_audit.json')
    assert a['physics_steps']==0
    for r in a['rows']:
        assert r['initial_contact_count']==0
        assert np.allclose(r['initial_actuator_force'],0)
        assert r['maximum_initial_hand_acceleration_radps2']>0
