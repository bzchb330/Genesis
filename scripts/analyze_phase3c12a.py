"""Render frozen Phase 3C-1.2A evidence; does not run simulation or search."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from seqgrasp import phase3c12a as c

FIG=c.ROOT/'docs/figures/phase3C12A'
BASE='6ca46034a743ba265b7ef58be452decdcb138f33'
BRANCH='codex/phase3C12a-contact-gravity-wrench-audit'
NAMES=['old_calibration_contact_identity','local_normal_vs_joint_approach',
 'corrected_normal_force_vs_command_offset','contact_normal_tracking',
 'object_weight_vs_calibrated_contact_force','MRL_contact_normal_cone',
 'MRL_gravity_vs_normal_cone','MRL_friction_dependence',
 'MRL_gravity_orientation_friction_map','transport_vs_storage_optimal_orientation',
 'old_ROLE_T_implementation_audit','true_thumb_assisted_topologies',
 'ROLE_T_true_contact_normal_cone','ROLE_T_true_gravity_orientation_map',
 'MRL_vs_ROLE_T_true_friction_requirement','index_middle_workspace_ROLE_T_true',
 'resource_recovery_kinematic_preliminary','phase3C12A_decision_summary']
SURFACES=['middle','ring','little','palm','thumb','index']


def read(name): return json.loads((c.OUTPUT/name).read_text())
def vec(value): return '['+', '.join(f'{x:.9g}' for x in value)+']'
def norm(value): return float(np.linalg.norm(value))
def codepath(function): return f'seqgrasp/phase3c12a.py:{inspect.getsourcelines(function)[1]}'


def summarize():
    old=read('calibration_autopsy.json'); cal=read('corrected_calibration.json')
    audits=read('mechanics_audits.json')['candidates']; search=read('true_role_t_search.json')
    work=read('true_role_t_workspace.json'); archived=c.read_old('resource_workspace_audit.json')
    m=audits[0]; t=audits[3]; b=m['best']; tb=t['best']; sol=b['actual_friction']; ts=tb['actual_friction']
    fractions=archived['retained_fraction']; q=b['configuration_rad']; g=b['gravity_direction_palm']
    validation=read('validation.json') if (c.OUTPUT/'validation.json').exists() else {'pytest':'NOT RUN YET','diff_check':'NOT RUN YET'}
    answers=[
      BRANCH,BASE,
      'Sphere geom phase3c07_sphere_geom paired with middle/ring/little/thumb/index distal collision_0 meshes and phase3_palm_rh_palm_collision_1 box; exact IDs, bodies, positions and normals in calibration_autopsy.json.',
      f"{old['switching_count']} switches in all 66 reconstructed old samples; 0 in 60 corrected branches.",
      'Yes: nonzero tangential contact-position migration on each surface. This alone does not explain all force nonmonotonicity.',
      'No: geom-centroid radial sphere translation, stationary hand joints/surfaces. Mostly normal (0.9603-0.9992 fraction), but not measured local-normal displacement; force was sampled before free settling.',
      'Material-point Cartesian target along runtime inward normal; analytic mj_jac bounded joint IK. Existing actuator targets and existing object weld, 50 steps x 0.002 s, last 10 samples summarized. No kinematic clamping. Some IK targets are unreachable and settling drifts.',
      str(cal['command_offsets_mm'])+' mm, frozen before sweep; command offset is not penetration.',
      'All six settled normal-force curves are [0,0,0,0,0,0,0,0,0,0] N. Contacts disappear, not usable zero-force preload curves. Initial contact forces are recorded separately.',
      f"{cal['weight_n']:.17g} N; compiled mass {cal['weight_n']/9.81:.17g} kg.",
      'Settled force/weight = 0 at all 60 samples; initial transients are not calibrated capacity. Weight is a scale, not an automatic sum-force success threshold.',
      'Parameterization/measurement error confirmed; no geom-switch bug. Corrected calibration also fails stable control, so physical conformity cannot yet be inferred.',
      '; '.join(x['surface']+' '+vec(x['inward_normal_palm']) for x in m['network']['contacts']),
      f"Positive hull of two inward normals; rank 2, planar span {m['normal_cone_generator_span_deg']:.9f} deg; zero 3D solid angle (not a broad frictionless volume).",
      str(m['original']['frictionless']['feasible']),f"{m['original']['cone']['angular_distance_deg']:.9f} deg.",
      f"Compiled elliptic condim-6 minimum-load solution rho_max={m['compiled_original']['rho_max']:.12g}, mean/median={m['compiled_original']['rho_mean']:.12g}. Restricted point-force model infeasible; rho there is undefined.",
      'Original: translational-only point-force F/F/F/F/F/F. Full compiled wrench F/F/F/F/T/T at scales [0,.1,.25,.5,.75,1]; minimum tested .75, not exact continuous onset. Optimized normal-only solution feasible at every scale, minimum 0. Full curve keeps spin/rolling coefficients fixed; scale=0 there is not the all-frictionless model.',
      vec(g),f'{q[0]:.12g} rad ({np.rad2deg(q[0]):.9g} deg)',f'{q[1]:.12g} rad ({np.rad2deg(q[1]):.9g} deg)',f'{q[2]:.12g} rad ({np.rad2deg(q[2]):.9g} deg)',
      f"{b['cone']['angular_distance_deg']:.9g} deg",str(b['frictionless']['feasible']),f"{sol['rho_max']}; exact normal-only witness, not the minimum-load frictional solution.",
      vec(sol['force_residual_n'])+f" N; norm={norm(sol['force_residual_n']):.9g} N",
      vec(sol['torque_residual_nm'])+f" N*m; norm={norm(sol['torque_residual_nm']):.9g} N*m",
      'Not necessarily at the identified static orientation; physical preload capacity and dynamic retention are not established.',
      'Old ROLE-T allowed thumb but required only any two nearby surfaces and closed two nearest digits. Actual ring+little topology reporting was correct. It did not test mandatory thumb support.',
      str(search['evaluated']),f"{search['real_thumb_contact_count']} geometrically; {search['true_preloaded_count']} with thumb+opposing positive initial normal force; none settled as receivers.",
      'Two preloaded static candidates: thumb+ring and thumb+little. Frozen geometry-selected representative: '+search['selected_candidate_id']+'.',
      '; '.join(x['surface']+' '+vec(x['inward_normal_palm']) for x in t['network']['contacts']),
      f"Rank {t['normal_cone_rank']}, planar span {t['normal_cone_generator_span_deg']:.9f} deg; zero 3D normal-cone solid angle.",
      'Original false; optimized true.',vec(tb['gravity_direction_palm']),vec(tb['configuration_rad'])+' rad (forearm_PS, WRJ1, WRJ2).',
      str(ts['rho_max'])+'; normal-only witness.',
      'Force '+vec(ts['force_residual_n'])+' N; torque '+vec(ts['torque_residual_nm'])+' N*m.',
      f"Both achieve rho=0. MRL normal span {m['normal_cone_generator_span_deg']:.6g} vs true-T {t['normal_cone_generator_span_deg']:.6g} deg; loads {sol['total_normal_force_n']:.9g} vs {ts['total_normal_force_n']:.9g} N. No demonstrated true-T superiority.",
      'Not demonstrated. Both need actuator-realizable preload. Full-wrench sample feasibility favors MRL in this bounded audit, not a morphology-wide ranking.',
      repr(fractions['thumb']),repr(fractions['index']),repr(fractions['opposition']),
      f"{work['role_t_true']['index']['reachable_volume_m3']:.17g} m^3; baseline {work['baseline']['index']['reachable_volume_m3']:.17g}; fraction {work['retained_fractions']['index']:.17g}.",
      f"{work['role_t_true']['middle']['reachable_volume_m3']:.17g} m^3; baseline {work['baseline']['middle']['reachable_volume_m3']:.17g}; fraction {work['retained_fractions']['middle']:.17g}.",
      f"Aperture-paired midpoint hull {work['role_t_true']['joint_acquisition']['opposition_midpoint_volume_m3']:.17g} m^3; fraction {work['retained_fractions']['joint_acquisition']:.17g}. Independent configurations, not true opposition or joint collision-free acquisition proof.",
      'Archived MRL preserves thumb/index resource fractions better than this true-T state preserves index/middle. Different samples and morphologies prevent a direct usefulness ranking; usefulness requires PI definition.',
      'CASE E: Cartesian calibration does not maintain stable, interpretable contacts with existing controls. D-type measurement correction and conditional A-type static orientation result are secondary.',
      'Invalid as a settled local-normal preload calibration; old instantaneous measurements are reproducible, not fabricated.',
      'Yes in the fixed-network static equations; no physical receiver claim. MRL original frictionless false -> optimized true.',
      'Conditionally as a static geometry, not validated storage.',
      'No superiority shown; actual thumb contact is now demonstrated only in static configurations.',
      'No actuator-realizable, calibrated receiver has been established.',
      'No receiver authorized for dynamics. Conditional candidate: ROLE_MRL_05 at the reported q, middle/little required normal forces '+str([r['normal_force_n'] for r in sol['forces']])+' N; preload capacity unestablished.',
      'No: debug quasi-static calibration first, then obtain PI authorization for Phase 3C-1.2B.',
      'No.', 'Yes.', 'Yes.', 'Yes.',
      'Bounded Phase 3C-1.2A calibration-control/geometry debugging, with fixed physics. Only after measured sustained preload: propose Phase 3C-1.2B direct-hold validation at the identified storage pose.',
      validation['pytest'],validation['diff_check'],f'{len(NAMES)} vector PDFs in docs/figures/phase3C12A/.',
      'docs/PHASE3C12A_RESULTS.md; docs/PHASE3C12A_CALIBRATION_AUTOPSY.md; docs/PHASE3C12A_ROLE_T_AUDIT.md; docs/PHASE3C_RESOURCE_RECOVERY_KINEMATIC_RESULT.md; outputs/phase3C12A/*.json; exact manifest below.'
    ]
    assert len(answers)==65
    inputs={p.relative_to(c.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (c.old.OUTPUT).glob('*.json')}
    summary=dict(phase='3C-1.2A',branch=BRANCH,base_commit=BASE,primary_case='E',answers={str(i+1):x for i,x in enumerate(answers)},
        contract=c.phase_contract(),old_artifact_sha256=inputs,archived_mrl_fractions=fractions,
        calibration_stable_capacity_established=False,mechanical_solutions_are_conditional=True,
        figures=[f'docs/figures/phase3C12A/{x}.pdf' for x in NAMES],
        machine_artifacts=sorted(p.relative_to(c.ROOT).as_posix() for p in c.OUTPUT.glob('*.json')))
    c.save('phase3c12a_results.json',summary)
    headings={1:'Calibration audit',13:'MRL mechanics',29:'True thumb-assisted storage',42:'Resource workspace',49:'Final decision'}
    lines=['# Phase 3C-1.2A final report','',
           'Primary outcome: **CASE E**. Static feasibility does not establish an actuator-realizable receiver. Phase 3C-1.1 was committed and both branches pushed; Phase 3C-1.2A remains uncommitted.','']
    for i,answer in enumerate(answers,1):
        if i in headings: lines+=['## '+headings[i],'']
        lines+=[f'{i}. {answer}','']
    lines+=['## Solver interpretation and limits','',
        'Two explicitly separated models are stored. The translational point-force model retains all six force/torque equations, using a 64-ray inscribed Coulomb cone (maximum radial conservatism 0.12046%). An outer polygon checks infeasibility certificates. It excludes spin/rolling moments. For two radial sphere contacts its force-feasible directions can have zero 3D angular volume. Never equate its failure with failure of the compiled condim-6 model.','',
        'The compiled-reference model uses the runtime elliptic condim-6 cone. A cutting-plane outer LP imposes exact Euclidean-cone separation until slack >= -1e-9 N; an infeasible outer LP certifies infeasibility. Moment coefficients retain their physical length units. rho_translation is reported separately from the combined elliptic-cone utilization. Full-reference objective is minimum total normal load, NOT minimum rho: its best-pose solution may use friction to reduce normal load even though a separate rho=0 witness exists.','',
        'Offline friction scales affect translational coefficients only; spin/rolling coefficients stay compiled. All-frictionless feasibility is separately computed with point normals and no contact moments. No offline coefficient is written to MuJoCo. No force-capacity constraint can be justified from the failed calibration, so static results assume unbounded compressive normal-load availability.','',
        'Orientation search: 45 compiled/diagnostic-range coarse poses plus three local refinements. Refinement minimizes normal-cone projection residual; descriptors are ranked lexicographically. This is not a global guarantee or a uniform solid-angle estimate. Sample counts include clustered refinement poses. The transport comparator is the first archived C07_STATE_00000 transport-optimal row, fixed in advance, not a universal transport optimum. All 18 frozen true-T attempts are additionally audited in local_candidate_mechanics.json: 15 lack the mandatory geometric network; all three geometric networks (06, 07, 08) admit optimized rho=0. Only 06 and 07 have positive initial forces. Selection remains frozen at 07; no new geometry search or receiver dynamics.','',
        '| Candidate | Normal span (deg) | Point-force feasible / 48 | Full compiled feasible / 48 | Frictionless / 48 | Transport-storage angle (deg) |',
        '|---|---:|---:|---:|---:|---:|']
    for a in audits:
        lines.append(f"| {a['network']['candidate_id']} | {a['normal_cone_generator_span_deg']:.6f} | {a['sampled_feasible_count']} | {a['sampled_compiled_feasible_count']} | {a['sampled_frictionless_count']} | {a['transport_comparison']['gravity_direction_difference_deg']:.6f} |")
    lines+=['','Each optimized pose preserves the measured geom pairs, palm-frame contact positions and normals under a rigid sphere/palm transform (static forward only). No world-gravity rotation. These checks do not establish a trajectory into that pose. Worst sampled configurations, all per-pose solutions, contact bodies/geoms/gaps/lever arms and force components are in mechanics_audits.json.','',
        'New preload representation is commanded Cartesian offset + resulting measured contact force. Nominal sustainable capacity remains **unknown**, not zero capacity: this experiment lost contact. No publication materiality or receiver-validity threshold was invented.','',
        'Contact sign follows the [official MuJoCo contact convention](https://mujoco.readthedocs.io/en/latest/XMLreference.html): normal points geom1 to geom2; flip when the object is geom1. Runtime dim=6 and friction=[0.5,0.5,0.01,0.003,0.003]. Closest-point and Jacobian calls follow the [official API](https://mujoco.readthedocs.io/en/3.2.6/APIreference/APIfunctions.html).','',
        '## Artifacts and reproduction','',
        '`scripts/run_phase3c12a.py` runs the frozen calibration/search; do not rerun for report generation. `scripts/analyze_phase3c12a.py` consumes existing data only. The machine summary hashes every existing Phase 3C-1.1 JSON. Generated datasets remain ignored under outputs/. PDFs are deliberate report artifacts.','']
    lines.extend('- '+p for p in summary['machine_artifacts']+summary['figures'])
    (c.ROOT/'docs/PHASE3C12A_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return old,cal,audits,search,work,archived,summary


def write_audits(old,cal,audits,search,work,archived,summary):
    lines=['# Phase 3C-1.2A calibration autopsy','',
        'All 66 original instantaneous contact-force values reconstruct exactly (maximum error 0 N). No old settling trajectories were rerun. The old hand surface and joints never moved; the sphere moved along a geom-centroid radial ray. Measured force came before the following 25 free-object steps, not after a controlled quasi-static normal command.','',
        'No active geom switching was found. Tangential contact migration is real but small; it does not prove the entire cause of force nonmonotonicity. Do not relabel reproducible instantaneous forces as corrupt data.','',
        '| Surface | Old normal fraction range | Maximum consecutive tangential migration (mm) | Corrected 0.2-mm IK error (mm) | Command normal fraction | Settled normal fraction | Settled drift (mm) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for surface in SURFACES:
        rows=[r for r in old['rows'] if r['surface']==surface]; fractions=[r['relative_object_motion']['normal_motion_fraction'] for r in rows if r['relative_object_motion']['normal_motion_fraction'] is not None]
        r=next(r for r in cal['rows'] if r['surface']==surface and r['command_offset_mm']==.2)
        lines.append(f"| {surface} | {min(fractions):.6f}-{max(fractions):.6f} | {max(r['contact_tangential_migration_m'] or 0 for r in rows)*1000:.6f} | {r['ik_error_m']*1000:.6f} | {r['commanded_motion']['normal_motion_fraction']:.6f} | {r['actual_settled_motion']['normal_motion_fraction']:.6f} | {r['actual_settled_motion']['tangential_drift_m']*1000:.6f} |")
    lines+=['','The local normal is measured using the frozen 0.025-mm runtime contact probe. A sphere tangent material point is targeted via bounded analytic Jacobian IK. Palm/root uses existing forearm/wrist joints. Targets that cannot be reached are retained and explicitly report IK residuals; they are not called exact normal motion.','',
        'Every sample then uses existing actuator position commands and the existing object fixture for 50 steps (100 ms). Final ten-step means are zero for all six surfaces because contact is lost. Large material-point drift shows that kinematic qpos realization does not imply actuator-held configuration. This failed settling trial is not a valid quasi-static force-capacity measurement. Further debugging must characterize joint-target realization/coupling and geometry/control drift without changing contact physics.','',
        'Geom identity, contact normals, contact distance, geometric closest-point distance, forces and per-step contact lists are stored. Branches terminate on a different contact geom. Absent contact distance is absent/null, not fabricated as a penetration. Nominal sustainable force capacity cannot be inferred.','',
        'Source paths: old seqgrasp/phase3c11.py:_tangent_setup and force_approach_calibration; new '+codepath(c.old_calibration_autopsy)+', '+codepath(c.corrected_calibration)+', '+codepath(c.cartesian_ik)+', '+codepath(c.contact_rows)+'.','',
        'Machine evidence: outputs/phase3C12A/calibration_autopsy.json and corrected_calibration.json. Preload means **normal command offset + actual resulting force**, not geometric penetration alone.']
    (c.ROOT/'docs/PHASE3C12A_CALIBRATION_AUTOPSY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    text='''# Phase 3C-1.2A old ROLE-T implementation audit

The old ROLE-T definition allowed thumb, ring, little and palm, but required only any two nearby storage surfaces. Closure targeted the two nearest digit surfaces. Thumb was neither mandatory in the prefilter nor mandatory in closure. It was not filtered out by an explicit thumb-exclusion rule; the selected target surfaces did not require it. All six old initializers reported ring+little with zero thumb normal force. The topology reporting was correct; the claim that this ruled out thumb support was not justified.

Code: seqgrasp/phase3c11.py:488 storage_role_mechanics_search (prefilter and mechanics), :284 local_preload_closure (nearest-target selection), and :446 _role_configurations. The new audit is in seqgrasp/phase3c12a.py:old_role_t_audit and is_true_role_t.

The bounded new search uses six archived seeds x three opposing supporters (ring, little, palm), with thumb mandatory and index/middle opened. It searches tangent geometry within a 15-mm center neighborhood and compiled joint bounds, then commands 0.1-mm local normal offsets via IK. No contact physics change or dynamics. Three of 18 candidates have real thumb plus opposing geometric contact; two have positive initial thumb+opposing forces. These are instantaneous static configurations, not stable preloaded receivers. Candidate ROLE_T_TRUE_07 (thumb+little) is selected by geometry and least overlap before orientation outcomes, not claimed globally optimal. Its full network is audited alongside three archived candidates. Failed calibration prevents a force-capacity-based search or receiver acceptance.

All 18 frozen attempts also receive an explicit audit disposition in local_candidate_mechanics.json. The three valid geometric networks (06, 07, 08) all admit optimized normal-only solutions; only 06 and 07 have positive initial forces. This supplementary offline audit does not change the frozen selected representative or run new geometry searches. No thumb-contact sample was silently reclassified as MRL. The side-by-side index/middle resource descriptor is kinematic and not evidence of second-object acquisition. Mandatory thumb enforcement repairs the old scientific comparison, but no superiority of true-T is established.

Machine evidence: outputs/phase3C12A/old_role_t_audit.json, true_role_t_search.json, mechanics_audits.json and true_role_t_workspace.json.
'''
    (c.ROOT/'docs/PHASE3C12A_ROLE_T_AUDIT.md').write_text(text,encoding='utf-8')
    f=archived['retained_fraction']
    text=f'''# Resource recovery: independent kinematic preliminary result

Frozen source: Phase 3C-1.1, commit {BASE}, outputs/phase3C11/resource_workspace_audit.json. Exact values are copied, not recalculated or selected from new outcomes.

| Preserved resource | Retained convex-hull volume fraction |
|---|---:|
| Thumb | {f['thumb']!r} |
| Index | {f['index']!r} |
| Thumb-index aperture-paired midpoint workspace | {f['opposition']!r} |

This result is kinematic/geometric. It does NOT demonstrate stable dynamic storage. No state passed the 200-step dynamic workspace gate in Phase 3C-1.1. It supports the premise that internal storage can preserve acquisition workspace **if a mechanically stable receiver can be realized**. Convex hulls can contain unreachable interiors; independent digit samples are not joint collision-free acquisition proofs.

The figure is docs/figures/phase3C12A/resource_recovery_kinematic_preliminary.pdf. Phase 3C-1.2A does not modify the archived measurements or infer object-B acquisition. Hashes of the old JSON artifacts are in outputs/phase3C12A/phase3c12a_results.json.
'''
    (c.ROOT/'docs/PHASE3C_RESOURCE_RECOVERY_KINEMATIC_RESULT.md').write_text(text,encoding='utf-8')


def render(old,cal,audits,search,work,archived,summary):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name,title,note):
        fig.suptitle(title,fontsize=16,x=.07,ha='left',y=.98)
        fig.text(.07,.025,note,fontsize=9,color='#4b5563',va='bottom')
        fig.subplots_adjust(left=.25 if name==NAMES[9] else .09,right=.96,top=.86,bottom=.19,wspace=.32,hspace=.55)
        fig.savefig(FIG/(name+'.pdf')); plt.close(fig)
    def small(): return plt.subplots(figsize=(10.8,6.2))
    colors=plt.cm.tab10(np.arange(6)); m,t=audits[0],audits[3]
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for ax,s in zip(axes.flat,SURFACES):
        rows=[r for r in old['rows'] if r['surface']==s]
        ax.plot([r['old_approach_mm'] for r in rows],[r['stored_force_n'] for r in rows],'o-')
        ax.set(title=s,xlabel='Old radial offset (mm)',ylabel='Instantaneous force (N)')
    save(fig,NAMES[0],'Old calibration: exact reconstruction, fixed geom identities','66 samples; maximum force reconstruction error 0 N. No switching. Not settled force curves.')
    fig,axes=plt.subplots(1,2,figsize=(11,6.2))
    for j,s in enumerate(SURFACES):
        r=next(r for r in cal['rows'] if r['surface']==s and r['command_offset_mm']==.2)
        axes[0].bar(j-.18,r['commanded_motion']['normal_motion_fraction'],.36,color='#2563eb')
        axes[0].bar(j+.18,r['actual_settled_motion']['normal_motion_fraction'],.36,color='#d97706')
        axes[1].bar(j,r['actual_settled_motion']['tangential_drift_m']*1000,color=colors[j])
    for ax in axes: ax.set_xticks(range(6),SURFACES,rotation=30)
    axes[0].set(title='Normal fraction: IK (blue), settled (orange)',ylabel='Absolute projection / motion',ylim=(0,1.1))
    axes[1].set(title='Settled material-point tangential drift',ylabel='Drift (mm)')
    save(fig,NAMES[1],'Cartesian commands do not imply actuator-held normal motion','Representative 0.20-mm command. Old calibration used sphere radial translation, not joint increments.')
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for ax,s in zip(axes.flat,SURFACES):
        rows=[r for r in cal['rows'] if r['surface']==s]
        ax.plot([r['command_offset_mm'] for r in rows],[r['mean_normal_force_n'] for r in rows],'o-')
        ax.set(title=s,xlabel='Normal command offset (mm)',ylabel='Tail force (N)',ylim=(-.005,.015))
        ax.text(.5,.75,'CONTACT LOST',transform=ax.transAxes,ha='center',color='#b91c1c')
    save(fig,NAMES[2],'Corrected command sweep: no sustained calibration contact','Frozen 10 offsets; existing actuator targets + existing object fixture; 100-ms settle, last 20-ms mean.')
    fig,axes=plt.subplots(1,2,figsize=(11,6.2))
    for j,s in enumerate(SURFACES):
        r=next(r for r in cal['rows'] if r['surface']==s and r['command_offset_mm']==.2)
        x=[v['step']*.002*1000 for v in r['timeline']]
        angles=[]
        for v in r['timeline']:
            con=next((z for z in v['contacts'] if r['target_geom_id'] in z['geom_ids']),None)
            angles.append(np.nan if con is None else np.rad2deg(np.arccos(np.clip(np.dot(con['inward_normal_world'],r['reference_normal_world']),-1,1))))
        axes[0].plot(x,angles,label=s,color=colors[j]); axes[1].plot(x,[z['normal_force_n'] for z in r['timeline']],label=s,color=colors[j])
    axes[0].set(xlabel='Settling time (ms)',ylabel='Normal change (deg)',title='Normal trace stops when contact is absent')
    axes[1].set(xlabel='Settling time (ms)',ylabel='Normal force (N)',title='Contact-force transients'); axes[1].legend(fontsize=8)
    save(fig,NAMES[3],'Fixed-pair contact-normal tracking at 0.20-mm command','No missing contact is interpolated. No switch was detected; all target contacts disappeared.')
    fig,ax=small(); initial=[]
    for s in SURFACES:
        r=next(r for r in cal['rows'] if r['surface']==s and r['command_offset_mm']==.2)
        initial.append(sum(x['normal_force_n'] for x in r['initial_contacts'])/cal['weight_n'])
    ax.bar(SURFACES,initial,color=colors); ax.axhline(1,color='black',ls='--',label='Object weight scale')
    ax.plot(SURFACES,np.zeros(6),'rx',ms=10,label='Settled (all zero)'); ax.legend()
    ax.set(ylabel='Contact force / object weight',title=f"Compiled sphere weight = {cal['weight_n']:.9f} N")
    save(fig,NAMES[4],'Initial transients are not sustained preload capacity','Weight is a physical reference, not a universal sum-normal-force success threshold. Offset 0.20 mm.')
    def cone(a,name,title):
        fig=plt.figure(figsize=(10.8,6.2)); ax=fig.add_subplot(111,projection='3d')
        normals=np.array([z['inward_normal_palm'] for z in a['network']['contacts']])
        for i,(n,z) in enumerate(zip(normals,a['network']['contacts'])): ax.quiver(0,0,0,*n,label=z['surface'],linewidth=2,color=['#2563eb','#0d9488'][i])
        for f in np.linspace(0,1,35):
            v=f*normals[0]+(1-f)*normals[1]; v/=np.linalg.norm(v); ax.plot([0,v[0]],[0,v[1]],[0,v[2]],color='#93c5fd',alpha=.4)
        v=-np.asarray(a['best']['gravity_direction_palm']); ax.quiver(0,0,0,*v,color='#dc2626',linewidth=3,label='Optimized -gravity')
        ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel='Palm X',ylabel='Palm Y',zlabel='Palm Z'); ax.legend(loc='upper left',bbox_to_anchor=(-.25,1))
        save(fig,name,title,f"Planar cone span {a['normal_cone_generator_span_deg']:.3f} deg; zero 3D solid angle. Conditional fixed-network mechanics.")
    cone(m,NAMES[5],'ROLE_MRL_05: middle + little normal-support cone')
    fig,ax=small(); vals=[m['original']['cone']['angular_distance_deg'],m['best']['cone']['angular_distance_deg']]
    ax.bar(['Original storage pose','Optimized reachable pose'],vals,color=['#d97706','#2563eb']); ax.set(ylabel='Angular distance to normal cone (deg)',ylim=(0,110))
    for i,v in enumerate(vals): ax.text(i,v+3,f'{v:.3f}',ha='center')
    save(fig,NAMES[6],'Storage gravity enters the MRL normal-support cone','NNLS projection; polar-cone cases use nearest nonzero cone direction, not the directionless origin.')
    fig,ax=small(); scales=c.config().mechanics['friction_scales']
    ax.plot(scales,[r['solution']['feasible'] for r in m['compiled_original_friction_curve']],'o-',label='Original, full compiled wrench')
    ax.plot(scales,np.ones(6),'s--',label='Optimized, normal-only witness')
    ax.plot(scales,np.zeros(6),'x:',label='Original, point-force-only model')
    ax.set(xlabel='Offline translational friction scale',ylabel='Feasible',yticks=[0,1],ylim=(-.1,1.3)); ax.legend(loc='upper left')
    save(fig,NAMES[7],'MRL friction dependence: distinguish point forces from contact moments','Full reference retains compiled spin/rolling capacity. All-frictionless feasibility is evaluated separately.')
    def gravitymap(a,name,title):
        fig,ax=small(); rows=a['rows']; g=np.asarray([r['gravity_direction_palm'] for r in rows]); azi=np.rad2deg(np.arctan2(g[:,1],g[:,0])); elev=np.rad2deg(np.arcsin(g[:,2])); valid=np.asarray([r['compiled_wrench']['feasible'] for r in rows])
        rho=[r['compiled_wrench']['rho_max'] for r in rows if r['compiled_wrench']['feasible']]
        sc=ax.scatter(azi[valid],elev[valid],c=rho,cmap='viridis',vmin=0,vmax=1,s=65,edgecolors='black',linewidths=.3)
        ax.scatter(azi[~valid],elev[~valid],marker='x',color='#b91c1c',s=55,label='Full-wrench infeasible')
        best=np.asarray(a['best']['gravity_direction_palm']); ax.scatter(np.rad2deg(np.arctan2(best[1],best[0])),np.rad2deg(np.arcsin(best[2])),marker='*',s=220,color='#ef4444',label='Normal-only witness')
        fig.colorbar(sc,ax=ax,label='Translation rho at minimum normal load'); ax.set(xlabel='Gravity azimuth in palm (deg)',ylabel='Gravity elevation (deg)'); ax.legend(fontsize=8)
        save(fig,name,title,f"{a['sampled_compiled_feasible_count']}/48 full-wrench feasible; {a['sampled_frictionless_count']}/48 frictionless. Nonuniform samples, not region volume.")
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,a in zip(axes.flat,audits):
        rows=a['rows']; g=np.asarray([r['gravity_direction_palm'] for r in rows])
        azi=np.rad2deg(np.arctan2(g[:,1],g[:,0])); elev=np.rad2deg(np.arcsin(g[:,2]))
        valid=np.asarray([r['compiled_wrench']['feasible'] for r in rows])
        rho=[r['compiled_wrench']['rho_max'] for r in rows if r['compiled_wrench']['feasible']]
        sc=ax.scatter(azi[valid],elev[valid],c=rho,cmap='viridis',vmin=0,vmax=1,s=35)
        ax.scatter(azi[~valid],elev[~valid],marker='x',color='#b91c1c',s=30)
        ax.set(title=f"{a['network']['candidate_id']} ({sum(valid)}/48 feasible)",xlabel='Gravity azimuth (deg)',ylabel='Gravity elevation (deg)')
        fig.colorbar(sc,ax=ax,label='Minimum-load translation rho')
    save(fig,NAMES[8],'Reachable gravity: MRL and all primary comparison networks','Red crosses: full-wrench infeasible. Each network has 3/48 normal-only witnesses. Nonuniform sample counts, not area.')
    fig,ax=small(); labels=[a['network']['candidate_id'].replace('ROLE_','') for a in audits]
    ax.barh(labels,[a['transport_comparison']['gravity_direction_difference_deg'] for a in audits],color='#2563eb'); ax.set(xlabel='Gravity-direction separation (deg)')
    save(fig,NAMES[9],'Transport and storage select different orientations','Transport comparator: archived C07_STATE_00000 optimum, fixed before comparison. No transport was executed.')
    fig,ax=small(); ax.axis('off')
    blocks=[('Declared resource role','Thumb allowed for storage; index/middle preserved.'),('Actual selection rule','Any two nearby surfaces; close the two nearest digits.'),('Measured old topology','Ring + little. Thumb force zero in all six selected states.'),('Correct interpretation','Mandatory thumb support was not tested. Reporting was accurate; rejection was not justified.')]
    for i,(head,body) in enumerate(blocks):
        y=.9-i*.24; ax.text(.02,y,head,weight='bold',fontsize=12); ax.text(.02,y-.08,body,fontsize=11)
    save(fig,NAMES[10],'Old ROLE-T: availability was mistaken for required participation','Implementation audit only; archived outcomes and scientific thresholds remain unchanged.')
    fig,ax=small(); counts=[search['evaluated'],search['real_thumb_contact_count'],search['true_preloaded_count'],0]
    ax.bar(['Local attempts','Real thumb + opponent','Positive initial forces','Validated receivers'],counts,color=['#64748b','#2563eb','#0d9488','#dc2626']); ax.set(ylabel='Count',ylim=(0,21))
    for i,v in enumerate(counts): ax.text(i,v+.4,str(v),ha='center')
    save(fig,NAMES[11],'Mandatory-thumb search: static contacts, not receivers','18 local attempts; static positive-force topologies thumb+ring and thumb+little. Selected ROLE_T_TRUE_07.')
    cone(t,NAMES[12],'ROLE_T_TRUE_07: thumb + little normal-support cone')
    gravitymap(t,NAMES[13],'True thumb-assisted state: reachable gravity and friction')
    fig,axes=plt.subplots(1,2,figsize=(11,6.2)); labels=['MRL','True-T']
    axes[0].bar(labels,[m['normal_cone_generator_span_deg'],t['normal_cone_generator_span_deg']],color=['#2563eb','#0d9488']); axes[0].set(ylabel='Planar generator span (deg)',title='Not 3D angular coverage')
    axes[1].bar(labels,[m['best']['actual_friction']['total_normal_force_n'],t['best']['actual_friction']['total_normal_force_n']],color=['#2563eb','#0d9488']); axes[1].axhline(cal['weight_n'],color='black',ls='--'); axes[1].set(ylabel='Required normal-only load (N)',title='Both have minimum rho = 0')
    save(fig,NAMES[14],'No demonstrated mechanical superiority for true-T','Loads assume available compressive capacity. Calibration has not established that capacity.')
    fig,ax=small(); f=work['retained_fractions']; ax.bar(['Index','Middle','Aperture-paired midpoint'],[f['index'],f['middle'],f['joint_acquisition']],color=['#2563eb','#0d9488','#64748b']); ax.set(ylabel='Retained hull-volume fraction',ylim=(0,1.12))
    for i,v in enumerate(f.values()): ax.text(i,v+.02,f'{100*v:.2f}%',ha='center')
    save(fig,NAMES[15],'True-T preserves index motion but limits middle resources','512 matched samples/finger. Independent kinematic aperture descriptor; no object B or acquisition proof.')
    fig,ax=small(); f=archived['retained_fraction']; ax.bar(['Thumb','Index','Thumb-index aperture workspace'],list(f.values()),color=['#2563eb','#0d9488','#64748b']); ax.set(ylabel='Retained hull-volume fraction',ylim=(0,1.15))
    for i,v in enumerate(f.values()): ax.text(i,v+.025,f'{100*v:.4f}%',ha='center')
    save(fig,NAMES[16],'Internal storage can preserve acquisition workspace geometrically','Frozen Phase 3C-1.1 exact values. Kinematic preliminary result; NOT stable dynamic storage.')
    fig,ax=small(); ax.axis('off')
    items=[('CASE E - current gate','Stable interpretable contact calibration has not been achieved.'),('Conditional static result','MRL and true-T admit normal-only support at reachable poses.'),('Not established','Actuator-realizable preload, robust storage, or true-T superiority.'),('Next authorized boundary','Debug calibration; propose direct-hold only after PI review.')]
    for i,(head,body) in enumerate(items):
        y=.9-i*.24; ax.text(.02,y,head,weight='bold',fontsize=12,color='#b91c1c' if i==0 else '#1e3a8a'); ax.text(.02,y-.085,body,fontsize=11)
    save(fig,NAMES[17],'Phase 3C-1.2A: do not cross the receiver-validation gate','No new physics, skin, handoff, shape dynamics, RL, object B, or receiver hold batch. Work uncommitted.')


if __name__=='__main__':
    data=summarize(); write_audits(*data); render(*data)
    print('Wrote four reports, machine summary, and 18 vector PDF figures.',flush=True)
