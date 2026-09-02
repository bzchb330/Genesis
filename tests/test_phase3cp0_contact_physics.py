"""Independent contact-bench contracts and honest gates for unapproved physics."""
import hashlib
import inspect
import json
import subprocess

import mujoco
import numpy as np
import pytest

from seqgrasp import contact_bench as b, phase3cp0 as p
from seqgrasp.physical_admissibility import EngineeringGates, diagnose


@pytest.fixture(scope='module')
def snapshot(): return p.read('current_physics.json')


def suite(): return json.loads((p.OUTPUT/p.config()['legacy_name']/'results.json').read_text())
def baseline(): return [r for r in suite()['rows'] if r['tag'].endswith('repeat_0')]


def test_exact_active_contact_physics_extraction(snapshot):
    assert snapshot['options']['timestep']==.002 and snapshot['solver_name']=='Newton'
    assert snapshot['sphere']['solref']==[.02,1.]
    for pair in snapshot['runtime_original_shallow_hand_pairs']:
        for key in ('friction','solref','solimp','dim'):
            assert pair[key]==snapshot['runtime_sphere_plane_contact'][key]


def test_sphere_mass_radius(snapshot):
    assert snapshot['sphere_radius_m']==.0125
    assert snapshot['sphere_mass_kg']==pytest.approx(4/3*np.pi*.0125**3*1000)


def test_sphere_weight(snapshot): assert snapshot['sphere_weight_n']==pytest.approx(.08025787482217676)


def test_isolated_scene_creation(snapshot):
    s=b.create_bench(snapshot)
    assert (s.model.nq,s.model.nv,s.model.nu,s.model.ntendon,s.model.neq,s.model.ngeom)==(7,6,0,0,0,2)
    assert s.data.qpos[2]==s.radius


def test_frozen_load_set():
    assert p.frozen_config()['bench']['loads_n']==[.01,.02,.05,.08025787482217676,.1,.134311598,.2,.3,.5,1.]


def test_load_ramp_total_not_extra_weight(snapshot):
    v,r,h=b.force_schedule(.1,.002,.4,4.)
    assert v[r-1]==.1 and np.all(np.diff(v[:r])>0) and h==2000
    assert b.smooth_ramp(0)==0 and b.smooth_ramp(.5)==.5 and b.smooth_ramp(1)==1
    s=b.create_bench(snapshot); assert s.data.xfrc_applied[s.body_id,2]==pytest.approx(snapshot['sphere_weight_n'])


def test_steady_state_measurement():
    r=baseline()[0]; trace=b.load_trace(p.ROOT/r['trace'])
    assert r['mean']['normal_force_n']==pytest.approx(np.mean(b.column(trace,'normal_force_n')[-200:]))
    assert r['variance']['overlap_m']==pytest.approx(np.var(b.column(trace,'overlap_m')[-200:]))


def test_force_deformation_monotonicity():
    rows=baseline()
    assert np.all(np.diff([r['mean']['overlap_m'] for r in rows])>0)
    assert np.all(np.diff([r['mean']['normal_force_n'] for r in rows])>0)


def test_delta_over_radius(snapshot):
    for r in baseline(): assert r['mean']['delta_over_radius']==pytest.approx(r['mean']['signed_overlap_m']/snapshot['sphere_radius_m'])


def test_load_unload_protocol():
    v,r,h=b.force_schedule(.3,.004,.4,4.,True)
    assert h==1000 and len(v)==2*(r+h) and v[-1]==0
    np.testing.assert_allclose(v[:r]+v[r+h:r+h+r],.3)


def test_energy_accounting_identity():
    for row in suite()['rows']:
        tr=b.load_trace(p.ROOT/row['trace'])
        residual=b.column(tr,'kinetic_energy_j')-b.column(tr,'external_work_j')-b.column(tr,'gravity_work_j')-b.column(tr,'contact_work_j')
        np.testing.assert_allclose(residual,b.column(tr,'work_balance_residual_j'),atol=1e-16)


def test_timestep_sensitivity_preserves_durations_and_minimum_hold(snapshot):
    for mult in (.5,1,2):
        s=b.create_bench(snapshot,timestep_multiplier=mult)
        v,r,h=b.force_schedule(.1,s.model.opt.timestep,.4,4.)
        assert h>=1000 and len(v)*s.model.opt.timestep==pytest.approx(4.4)


def test_solver_sensitivity(snapshot):
    s=b.create_bench(snapshot,solver_diagnostic=True)
    assert s.model.opt.iterations==400 and s.model.opt.tolerance==1e-12
    assert s.model.opt.timestep==snapshot['options']['timestep']


def test_primary_bench_has_no_hand_controller_dependency():
    source=inspect.getsource(b)
    assert 'phase3c12b' not in source and 'build_shadow' not in source and 'actuator_target' not in source


def test_no_task_based_parameter_tuning(): assert not p.scope()['task_outcome_parameter_selection']


def test_fixed_friction_in_candidates(snapshot):
    for option in p.config()['candidate_options']:
        s=b.create_bench(snapshot,option); s.data.qpos[2]-=1e-8; mujoco.mj_forward(s.model,s.data)
        assert b.runtime_contact(s)['friction']==snapshot['runtime_sphere_plane_contact']['friction']


def test_optional_hertz_requires_all_constants():
    with pytest.raises(ValueError,match='NOT SPECIFIED'): b.hertz_force(.001,.0125)


def test_no_invented_elastic_constants():
    material=p.config()['material']
    assert all(v is None for k,v in material.items() if k!='provenance')
    assert p.read('material_audit.json')['hits']==[]


def test_versioned_physics_no_silent_promotion():
    cfg=p.config()
    assert cfg['legacy_name']=='LEGACY_PHASE3C_CONTACT_PHYSICS'
    assert cfg['approved_revised_physics'] is None
    with pytest.raises(PermissionError): p.require_validated_physics('CP-C')


def test_legacy_physics_and_artifacts_preserved():
    for name,sha in p.read('preserved_phase3C12B_hashes.json').items():
        assert hashlib.sha256((p.ROOT/name).read_bytes()).hexdigest()==sha


def test_fixed_sphere_hand_primitive_is_gated_not_fabricated():
    for label in ('CP-B','CP-C','CP-D'):
        with pytest.raises(PermissionError): p.require_validated_physics(label,{'pi_approved':False})
    p.require_validated_physics('CP-A')
    assert p.config()['hand_primitive']['steps']==1000


def test_desired_actual_force_logging_independent_bench():
    tr=b.load_trace(p.ROOT/baseline()[0]['trace'])
    assert {'target_load_n','normal_force_n','signed_overlap_m','vz_mps'} <= set(tr['fields'])
    assert p.config()['hand_primitive']['desired_forces_n']==[.01,.02,.05,.08,.1,.15,.2]


def raw(**kwargs):
    values=dict(radius_m=.0125,weight_n=.08,penetration_m=.001,normal_forces_n=[.03,.04],
                contact_gravity_force_n=[0,0,-.01],torque_nm=[0,0,0])
    values.update(kwargs); return diagnose(**values)


def test_actuator_saturation_diagnostic():
    d=raw(actuator_saturation_fraction=[.2,1.])
    assert d['maximum_actuator_saturation_fraction']==1


def test_penetration_logging():
    d=raw(); assert d['maximum_penetration_m']==.001 and d['penetration_radius_ratio']==.08


def test_physical_admissibility_has_no_default_publication_gate():
    d=raw(); assert d['engineering_gate_passed'] is None and not d['scientifically_validated']
    d=raw(gates=EngineeringGates('test-only',max_penetration_m=.0005))
    assert not d['engineering_gate_passed'] and 'MAX_PENETRATION_M' in d['violations']


def test_old_deep_receiver_rejected_without_simulation():
    r=p.read('legacy_regression.json')
    assert r['rejected'] and r['settled_maximum_penetration_m']==pytest.approx(.006598902740463493)
    assert r['peak_penetration_m']==pytest.approx(.008333245,abs=1e-9)
    assert r['last_sum_normal_force_n']==pytest.approx(6.18780922259)
    assert r['settled_mean_residual_force_n']==pytest.approx(2.64862017)
    assert not r['last_state']['engineering_gate_passed']
    assert not r['settled_peak_state']['engineering_gate_passed']
    assert r['settled_peak_state']['maximum_penetration_m']==r['settled_maximum_penetration_m']


def test_no_receiver_search(): assert not p.scope()['receiver_search'] and not p.scope()['B03_search']
def test_no_weld_release(): assert not p.scope()['receiver_weld_release']
def test_no_handoff(): assert not p.scope()['handoff']
def test_no_object_b(): assert not p.scope()['object_B']
def test_no_rl(): assert not p.scope()['RL']
def test_no_shape_study(): assert not p.scope()['shape_study']
def test_no_skin(): assert not p.scope()['skin']


def test_phase3c12b_backward_compatibility():
    diff=subprocess.check_output(['git','diff',p.BASE,'--','seqgrasp/phase3c12b.py','configs/phase3C12B_weld_release_receiver.yaml','docs/figures/phase3C12B'],cwd=p.ROOT,text=True)
    assert diff==''


def test_repeatability_saved_trajectories():
    for r in baseline():
        a=b.load_trace(p.ROOT/r['trace'])['samples']
        other=r['trace'].replace('repeat_0','repeat_1')
        np.testing.assert_array_equal(a,b.load_trace(p.ROOT/other)['samples'])


def test_positive_loaded_contact_and_no_negative_force():
    for r in baseline():
        assert r['contact_loss_transitions']==0 and r['negative_normal_force_samples']==0
        assert r['settling_after_ramp_seconds'] is not None


def test_nonfinite_input_cannot_be_hidden_by_clamping():
    d=raw(penetration_m=float('nan'),gates=EngineeringGates('test-only',max_penetration_m=.003))
    assert not d['engineering_gate_passed'] and 'NONFINITE_DIAGNOSTIC' in d['violations']
    d=raw(actuator_saturation_fraction=[.2,float('nan')])
    assert 'NONFINITE_DIAGNOSTIC' in d['violations']
