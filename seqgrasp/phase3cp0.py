"""P0 orchestration: compile-only hand audit, independent benches, saved-state regression.

No receiver dynamics and no hand control without the explicit protocol gate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess

import mujoco
import numpy as np
import yaml

from .config import ROOT
from . import contact_bench as bench
from .physical_admissibility import EngineeringGates, diagnose

OUTPUT=ROOT/'outputs/phase3CP0'
BASE='07a61ca7731b47f5b7bb03263cd956ad5c588b3e'
BRANCH='codex/phase3CP0-contact-physics-validation'


def config(): return yaml.safe_load((ROOT/'configs/phase3CP0_contact_physics.yaml').read_text())
def save(name,value):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return value
def read(name): return json.loads((OUTPUT/name).read_text(encoding='utf-8'))


def scope():
    return dict(receiver_search=False,B03_search=False,receiver_weld_release=False,handoff=False,
                object_B=False,RL=False,trajectory_optimization=False,shape_study=False,skin=False,
                production_physics_changed=False,task_outcome_parameter_selection=False)


def extract_current_physics():
    from . import phase3c12b as previous
    s=previous.build(); m=s.model; oid=previous.a._object_geom_id(s)
    def geom(i):
        r=dict(id=int(i),name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i))
        for k in ('type','size','friction','condim','priority','solref','solimp','margin','gap','solmix'):
            r[k]=getattr(m,'geom_'+k)[i].tolist()
        r['type_name']=mujoco.mjtGeom(r['type']).name
        return r
    hand=[geom(i) for ids in s.collision_geoms.values() for name in ids
          for i in [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)]]
    options={}
    for k in ('timestep','integrator','solver','iterations','tolerance','cone','impratio','ls_iterations',
              'ls_tolerance','noslip_iterations','noslip_tolerance','ccd_iterations','ccd_tolerance',
              'disableflags','enableflags'):
        v=getattr(m.opt,k); options[k]=v.tolist() if hasattr(v,'tolist') else v
    sphere=geom(oid); representative=next(r for r in hand if 'mfproximal' in r['name'])
    mass=float(m.body_mass[s.object_body_id]); radius=float(m.geom_size[oid,0])
    snap=dict(base_commit=BASE,mujoco_version=mujoco.__version__,options=options,
              integrator_name={0:'Euler',1:'RK4',2:'implicit',3:'implicitfast'}[m.opt.integrator],
              solver_name={0:'PGS',1:'CG',2:'Newton'}[m.opt.solver],cone_name={0:'pyramidal',1:'elliptic'}[m.opt.cone],
              sphere=sphere,representative_hand_geom=representative,hand_collision_geoms=hand,
              sphere_radius_m=radius,sphere_mass_kg=mass,sphere_inertia_kgm2=m.body_inertia[s.object_body_id].tolist(),
              effective_sphere_density_kg_m3=mass/(4/3*np.pi*radius**3),sphere_weight_n=mass*9.81,
              native_hand_fingerprint=previous.physics_fingerprint(s),explicit_contact_pair_count=int(m.npair),
              provenance=dict(timestep='configs/phase3_shadow_hand.yaml via seqgrasp/phase3/model.py',
                cone_impratio='Shadow Hand Menagerie right_hand.xml option',
                hand_solref_solimp='Menagerie plastic default class',sphere_solref_solimp='MuJoCo defaults',
                sphere_friction_condim_priority='project object config and seqgrasp/phase3/model.py',
                radius_density='seqgrasp/phase3c07.py: SPHERE_RADIUS_M and SPHERE_DENSITY_KG_M3',
                collision_material='plastic is an XML class label; visual materials do not provide elastic constants',
                integrator_solver_iterations_tolerance='MuJoCo defaults',margin_gap='MuJoCo defaults'))
    # Verify the actual combined pair in a shallow, compile-only probe, no dynamics.
    probe=bench.create_bench(snap); probe.data.qpos[2]-=1e-8; mujoco.mj_forward(probe.model,probe.data)
    snap['runtime_sphere_plane_contact']=bench.runtime_contact(probe)
    # Static source geometry only; never use the C12B deep-overlap geometry for interpretation.
    source,_=previous.receiver_setup(); runtime=[]
    for i in range(source.data.ncon):
        c=source.data.contact[i]
        if oid in (c.geom1,c.geom2):
            runtime.append(dict(dim=int(c.dim),friction=c.friction.tolist(),solref=c.solref.tolist(),solimp=c.solimp.tolist(),
                                distance_m=float(c.dist)))
    snap['runtime_original_shallow_hand_pairs']=runtime
    snap['pair_rule']='Sphere priority 1 exceeds hand/plane priority 0: sphere condim/friction/solref/solimp win; margins/gaps add.'
    return snap


def audit_materials():
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    pattern=re.compile(r"young.?s|poisson|elastic modulus|coefficient of restitution|sphere material|surface compliance",re.I)
    hits=[]; count=0
    for name in files:
        if Path(name).suffix.lower() not in ('.md','.yaml','.xml','.py','.txt'): continue
        count+=1
        for number,line in enumerate((ROOT/name).read_text(encoding='utf-8',errors='replace').splitlines(),1):
            if pattern.search(line): hits.append(dict(file=name,line=number,text=line.strip()))
    return dict(status='MATERIAL ELASTIC CONSTANTS NOT SPECIFIED BY PI',baseline_text_files_scanned=count,
                query=pattern.pattern,hits=hits,young_moduli_pa=None,poisson_ratios=None,
                restitution=None,physical_surface_compliance=None,
                damping_note='Numerical solref damping ratio and passive joint damping are specified, not material damping data.',
                sphere_note='Density=1000 kg/m^3 and display colors do not identify a material.',
                hand_note='Menagerie plastic class and visual materials do not specify a constitutive law.')


def run_audit():
    if (OUTPUT/'current_physics.json').exists(): raise FileExistsError('Audit exists; preserve rather than overwrite')
    snapshot=extract_current_physics(); save('current_physics.json',snapshot); save('material_audit.json',audit_materials())
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'outputs/phase3C12B').rglob('*') if p.is_file()}
    save('preserved_phase3C12B_hashes.json',hashes)
    frozen=dict(config=config(),config_sha256=hashlib.sha256((ROOT/'configs/phase3CP0_contact_physics.yaml').read_bytes()).hexdigest(),
                current_snapshot_sha256=hashlib.sha256((OUTPUT/'current_physics.json').read_bytes()).hexdigest(),
                frozen_before_bench_outcomes=True,scope=scope(),base_commit=BASE)
    return save('frozen_protocol.json',frozen)


def frozen_config():
    frozen=read('frozen_protocol.json')
    if config()!=frozen['config']: raise ValueError('Protocol changed after freeze')
    return frozen['config']


def run_suite(name,candidate=None):
    cfg=frozen_config(); protocol=cfg['bench']; snapshot=read('current_physics.json')
    if name!=cfg['legacy_name']:
        decision=read('legacy_classification.json')
        if decision['classification'] not in ('CP-B','CP-C','CP-D'): raise ValueError('Candidate study not authorized by classification')
        if candidate not in cfg['candidate_options']: raise ValueError('Unfrozen contact option')
    dest=OUTPUT/name; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'results.json').exists(): raise FileExistsError('Completed suite already exists')
    plan=[]
    for j,load in enumerate(protocol['loads_n']):
        for repeat in range(protocol['repeats']): plan.append((f'load_{j:02d}_repeat_{repeat}',load,1.,False,False))
    for j,load in enumerate(protocol['cycle_loads_n']): plan.append((f'cycle_{j:02d}',load,1.,False,True))
    for j,load in enumerate(protocol['sensitivity_loads_n']):
        for mult in (.5,2.): plan.append((f'dt_{mult}_{j:02d}',load,mult,False,False))
        plan.append((f'solver_{j:02d}',load,1.,True,False))
    rows=[]
    for tag,load,mult,solver,cycle in plan:
        path=dest/(tag+'.npz'); meta=dest/(tag+'.json')
        if path.exists() or meta.exists():
            if not (path.exists() and meta.exists()): raise RuntimeError('Partial artifact requires inspection; not overwritten')
            rows.append(json.loads(meta.read_text())); continue
        scene=bench.create_bench(snapshot,candidate,mult,solver)
        run=bench.run_bench(scene,load,protocol,cycle); result=bench.summarize(run,load,protocol,cycle)
        result.update(tag=tag,timestep_s=scene.model.opt.timestep,timestep_multiplier=mult,solver_diagnostic=solver,
                      trace=path.relative_to(ROOT).as_posix(),name=name)
        bench.save_trace(path,run); meta.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        rows.append(result)
        print(f'{name} {tag}: delta={result["mean"]["overlap_m"]*1000:.6g} mm, Fn={result["mean"]["normal_force_n"]:.9g} N',flush=True)
    value=dict(name=name,candidate=candidate,rows=rows,trials=len(rows),physics_scope='isolated bench only',
               load_convention='Total compressive load F; xfrc_applied_z=mg-F; gravity is retained',scope=scope())
    (dest/'results.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return value


def require_validated_physics(classification,approved_version=None):
    if classification!='CP-A' and not (approved_version and approved_version.get('pi_approved') is True):
        raise PermissionError('Hand force primitive blocked: CP-A or PI-approved versioned physics is required')


def legacy_regression():
    from . import phase3c12b as old
    summary=old.read('phase3c12b_summary.json'); rows=old.load_series(old.read('receiver_construction.json')['timeseries'])
    cfg=frozen_config(); gate=EngineeringGates(**cfg['legacy_regression_engineering_gate'])
    def diagnostic(row):
        v=np.asarray(row['sphere_linear_velocity_world_mps']); w=np.asarray(row['sphere_angular_velocity_world_radps'])
        snapshot=read('current_physics.json'); ke=.5*snapshot['sphere_mass_kg']*float(v@v)+.5*snapshot['sphere_inertia_kgm2'][0]*float(w@w)
        return diagnose(radius_m=.0125,weight_n=summary['weight_n'],penetration_m=row['maximum_penetration_m'],
                        normal_forces_n=[c['normal_force_n'] for c in row['contacts']],
                        contact_gravity_force_n=row['free_net_force_world_n'],torque_nm=row['free_net_torque_world_nm'],
                        actuator_saturation_fraction=row['actuator_saturation_fraction'],
                        external_support_force_n=row['weld_force_world_n'],external_support_torque_nm=row['weld_torque_world_nm'],
                        kinetic_energy_j=ke,environment_support=bool(np.linalg.norm(row['environment_force_world_n'])>0),gates=gate)
    result=dict(last_state=diagnostic(rows[-1]),peak_state=diagnostic(max(rows,key=lambda r:r['maximum_penetration_m'])),
                settled_peak_state=diagnostic(max(rows[-100:],key=lambda r:r['maximum_penetration_m'])),
                settled_maximum_penetration_m=summary['tail_maximum_penetration_m'],peak_penetration_m=summary['maximum_penetration_m'],
                last_sum_normal_force_n=summary['realized_last_normal_load_n'],
                settled_mean_residual_force_n=summary['mean_free_net_force_norm_n'],
                rejected=True,source='Saved Phase3C12B artifacts only; zero simulation steps',
                interpretation='Deep-overlap contact identities/normals/rho are not admissible evidence for topology, feasibility or morphology.',
                flags=['SETTLED_DEEP_OVERLAP','PEAK_DEEP_OVERLAP','HIGH_INTERNAL_LOAD_REPORTED','EXTERNAL_SUPPORT_CANCELS_LARGE_RESIDUAL'],
                scope_note='Only penetration uses an inherited numerical gate. Force/wrench flags expose raw scales without inventing scientific cutoffs.')
    return save('legacy_regression.json',result)
