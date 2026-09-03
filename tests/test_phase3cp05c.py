"""P0.5C compiled static-audit contracts and mandatory CASE-C stop."""
import hashlib
import mujoco
import numpy as np
import pytest
from seqgrasp import phase3cp05c as p, contact_physics as physics


@pytest.fixture(scope='module')
def result(): return p.read('outputs/phase3CP05C/summary.json')


def test_exact_nominal_and_settled_sources(result):
    assert result['nominal']['qpos'][:25]==p.read(p.config().nominal_source)['qpos'][:25]
    settled=p.read(p.config().settled_source)
    assert result['settled_cache_sha256']==settled['state_sha256']


def test_complete_joint_and_actuator_audits(result):
    assert len(result['joints'])==25 and len(result['joints_sorted'])==25 and len(result['actuators'])==21
    assert result['joints_sorted'][0]['joint_name']=='rh_RFJ2'
    assert result['joints_sorted'][0]['absolute_delta_q']==pytest.approx(.4998960959097237)


def test_compiled_classification_not_name_inference(result):
    row=next(x for x in result['joints'] if x['joint_name']=='rh_RFJ2')
    assert not row['directly_actuated'] and row['tendon_coupled'] and not row['independently_commandable']
    assert row['moment_terms']=={'rh_A_RFJ0':1.}


def test_moment_matrix_orientation_and_forward_mapping(result):
    A=np.asarray(result['moment_matrix']); assert A.shape==(21,25)
    assert result['mapping_verification']['matrix_shape']==[21,25]
    assert result['mapping_verification']['maximum_error']<1e-14
    assert result['mapping_verification']['maximum_probe_error']<1e-14


def test_inverse_static_force_and_forward_inverse_consistency(result):
    assert result['nominal']['max_acceleration']==pytest.approx(81.61685063369048,abs=1e-10)
    assert result['static_inverse_identity_error']<1e-14
    assert result['forward_inverse_consistency_max_error']<1e-14
    assert not np.any(result['qfrc_applied']) and not np.any(result['xfrc_applied'])


def test_unbounded_rank_singular_values_and_residual(result):
    a=result['allocation']; assert a['shape']==[21,25] and a['rank']==21
    assert len(a['singular_values'])==21 and a['missing_generalized_directions']==4
    assert a['unbounded']['norm']==pytest.approx(.005062838282265401)
    assert a['unbounded']['relative_residual']==pytest.approx(.009869084186549809)


def test_bounded_allocation_uses_existing_limits(result):
    a=result['allocation']; assert a['bounded_solver_success']
    assert np.allclose(a['bounded']['forces'],a['unbounded']['forces'],rtol=0,atol=1e-14)
    assert max(a['bounded']['utilization'])==pytest.approx(.10126823921670695)
    assert a['bounded']['saturated_indices']==[]


def test_residual_attribution_to_coupled_pairs(result):
    top=sorted(result['joints'],key=lambda x:-abs(x['unbounded_residual']))[:8]
    assert all(x['tendon_coupled'] and not x['independently_commandable'] for x in top)
    assert {x['joint_name'] for x in top}=={'rh_FFJ1','rh_FFJ2','rh_MFJ1','rh_MFJ2','rh_RFJ1','rh_RFJ2','rh_LFJ1','rh_LFJ2'}


def test_case_classification_mechanics(result):
    assert result['allocation']['case']=='CASE_C'
    A=np.eye(2); tau=np.array([.1,.2]); a=p.allocation(A,tau,np.array([-1.,-1.]),np.array([1.,1.])); assert a['case']=='CASE_A'
    b=p.allocation(A,np.array([2.,0.]),np.array([-1.,-1.]),np.array([1.,1.])); assert b['case']=='CASE_B'
    c=p.allocation(np.array([[1.,1.]]),np.array([1.,-1.]),np.array([-10.]),np.array([10.])); assert c['case']=='CASE_C'


def test_compiled_affine_actuator_model(result):
    m,d,_=p.build_nominal(); gain,bias=p.affine_model(m,d)
    assert np.all(gain>0) and np.allclose(bias,-gain*d.actuator_length)
    assert all(x['gaintype']=='mjGAIN_FIXED' and x['biastype']=='mjBIAS_AFFINE' and x['dyntype']=='mjDYN_NONE' for x in result['actuators'])


def test_case_c_forbids_ctrl_and_dynamics(result):
    with pytest.raises(PermissionError): p.require_case_a(result['allocation'])
    with pytest.raises(PermissionError): p.equilibrium_ctrl(*p.build_nominal()[:2],result['allocation'])
    assert not result['preload_constructed'] and result['direct_test'] is None and result['local_perturbation'] is None


def test_no_protected_model_changes(result):
    protocol=p.read('outputs/phase3CP05C/protocol.json'); m,_,_=p.build_nominal()
    assert p.immutable_parameters(m)==protocol['protected_parameters']
    assert not np.any(m.body_gravcomp)
    d=mujoco.MjData(m); assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)


def test_imp99_exact_settings_unchanged():
    v=physics.version(physics.IMP99)
    assert v['solref']==[.02,1.] and v['solimp']==[.99,.99,.001,.5,2]
    assert v['pair_friction']==[.5,.5,.01,.003,.003] and v['condim']==6 and v['timestep']==.002


def test_pi_override_recorded_but_no_v1_created(result):
    assert result['pi_override_recorded'] and p.config().pi_decision['simultaneous_mrl_not_required_for_contact_physics']
    assert not result['physics_v1_created'] and not (p.ROOT/'configs/PHYSICS_V1_NEAR_RIGID.yaml').exists()
    assert 'identical state and protocol once' in p.config().pi_decision['first_natural_sustained_mrl_recheck']


@pytest.mark.parametrize('key',['receiver','b03','flyby','bounded_force','handoff','object_B','rl','shape','skin'])
def test_out_of_scope_stays_gated(key): assert p.read('outputs/phase3CP05C/protocol.json')['scope'][key] is False


def test_resource_fractions_preserved_no_rescan(result):
    assert result['resource_fractions']==dict(thumb=.9559782183972225,index=1.,opposition=.9665998246424643)
    assert result['physics_steps']==0


def test_p05r_backward_compatibility():
    protocol=p.read('outputs/phase3CP05C/protocol.json')
    for path,sha in protocol['preserved_p05r_outputs'].items(): assert hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==sha
