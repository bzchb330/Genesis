"""Offline summaries/figures for executed C12B data. No simulation or search."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from seqgrasp import phase3c12b as b

ROOT=b.ROOT
FIG=ROOT/'docs/figures/phase3C12B'
FIGURES=['shadow_hand_actuation_chain','actuator_command_to_joint_response',
 'fixed_sphere_little_preload','fixed_sphere_middle_ring_little_force_curves',
 'virtual_target_preload_principle','actuator_force_and_saturation',
 'welded_MRL_receiver_construction','simultaneous_MRL_contact_force_timeline',
 'welded_contact_topology_timeline','counterfactual_free_sphere_net_force_before_release',
 'counterfactual_free_sphere_net_torque_before_release','weld_force_vs_hand_contact_force',
 'weld_release_sphere_motion','post_release_contact_force_timeline','post_release_net_force',
 'post_release_net_torque','modeled_vs_realized_contact_network','modeled_vs_realized_contact_normals',
 'modeled_vs_realized_friction_utilization','phase3C12B_causal_summary']
BASE='6f550dede4b94b3e755bb3b1b208a1880a21a562'
BRANCH='codex/phase3C12b-weld-release-receiver'


def norm(x): return float(np.linalg.norm(x))
def vec(x): return '['+', '.join(f'{v:.9g}' for v in x)+']'
def surface_force(row,s): return sum(c['normal_force_n'] for c in row['contacts'] if c['surface']==s)


def summarize():
    actuators=b.read('actuation_audit.json'); primitives=b.read('fixed_sphere_primitives.json')
    protocol=b.read('receiver_protocol.json'); construction=b.read('receiver_construction.json'); release=b.read('release_results.json')
    rows=b.load_series(construction['timeseries']); tail=rows[-100:]; source=b.source_receiver(); comparison=[]
    scene,_=b.receiver_setup(); scene.data.qpos[:]=rows[-1]['qpos']; mujoco.mj_forward(scene.model,scene.data)
    _,rotation=b.palm_transform(scene)
    for c in rows[-1]['contacts']:
        c['inward_normal_palm']=(rotation.T@c['inward_normal_world']).tolist()
        old=next((x for x in source['network']['contacts'] if x['geom_ids']==c['geom_ids']),None)
        comparison.append(dict(geom_names=c['geom_names'],surface=c['surface'],predicted_pair=old is not None,
            normal_palm=c['inward_normal_palm'],normal_difference_deg=None if old is None else float(np.rad2deg(np.arccos(np.clip(np.dot(c['inward_normal_palm'],old['inward_normal_palm']),-1,1)))),
            position_difference_palm_m=None if old is None else norm(rotation.T@(np.asarray(c['position_world_m'])-scene.data.xpos[scene.object_body_id])-old['lever_arm_palm_m']),
            normal_force_n=c['normal_force_n'],tangential_force_n=c['tangential_force_n'],rho_translation=c['rho_translation'],distance_m=c['distance_m']))
    group={}
    for s in ('middle','ring','little','palm'):
        values=[surface_force(r,s) for r in tail]; group[s]=dict(mean_n=float(np.mean(values)),variance_n2=float(np.var(values)),minimum_n=float(np.min(values)),maximum_n=float(np.max(values)))
    curve_details=[]
    for r in primitives['rows']:
        samples=b.load_series(r['timeseries']); observed=[c for row in samples for c in row['contacts'] if r['intended_geom_id'] in c['geom_ids']]
        first=observed[0] if observed else None; last=observed[-1] if observed else None
        drift=None if first is None else b.a.motion_components(np.asarray(last['position_world_m'])-first['position_world_m'],np.asarray(first['inward_normal_world']))['tangential_drift_m']
        curve_details.append(dict(**r,tangential_contact_migration_m=drift,
            last_observed_contact_distance_m=None if last is None else last['distance_m'],
            last_observed_normal_world=None if last is None else last['inward_normal_world'],
            final_contact_present=any(r['intended_geom_id'] in c['geom_ids'] for c in samples[-1]['contacts'])))
    mean_force=np.mean([r['free_net_force_world_n'] for r in tail],axis=0); mean_torque=np.mean([r['free_net_torque_world_nm'] for r in tail],axis=0)
    weight=b.a.object_weight(scene); selected=[]
    for s in ('little','palm','middle','ring'):
        target=protocol['choices'].get(s,{'offset':.05})['offset']
        selected.append(next(r for r in curve_details if r['surface']==s and r['virtual_offset']==target))
    cone=b.a.normal_cone([r['normal_palm'] for r in comparison],-rotation.T@scene.model.opt.gravity*scene.model.body_mass[scene.object_body_id])
    summary=dict(base_commit=BASE,branch=BRANCH,classification='RECEIVER_CONSTRUCTION_FAILURE',post_release_classification=None,
        setup_diagnostic_namespace='outputs/phase3C12B',primary_namespace='outputs/phase3C12B/fixed_support',
        scope_note='Inherited soft-weld debug: 32 primitives + one welded construction, zero releases. Explicit fixed-support definition: same 32 offsets + one deterministic construction, zero releases. No post-failure release batch or new receiver basin.',
        temporary_support=b.config().temporary_support,hand_contact_physics_unchanged=construction['physics_unchanged'],
        release_trials=release['trial_count'],weight_n=weight,selected_primitives=selected,primitive_curves=curve_details,
        receiver_normal_force_tail=group,receiver_last_contacts=comparison,realized_normal_cone=cone,
        mean_free_net_force_world_n=mean_force.tolist(),mean_free_net_force_norm_n=norm(mean_force),mean_free_net_force_weight_ratio=norm(mean_force)/weight,
        mean_free_net_torque_world_nm=mean_torque.tolist(),mean_free_net_torque_norm_nm=norm(mean_torque),
        mean_hand_force_world_n=np.mean([r['hand_force_world_n'] for r in tail],axis=0).tolist(),
        mean_hand_torque_world_nm=np.mean([r['hand_torque_world_nm'] for r in tail],axis=0).tolist(),
        mean_weld_force_world_n=np.mean([r['weld_force_world_n'] for r in tail],axis=0).tolist(),
        mean_weld_torque_world_nm=np.mean([r['weld_torque_world_nm'] for r in tail],axis=0).tolist(),
        maximum_penetration_m=max(r['maximum_penetration_m'] for r in rows),tail_maximum_penetration_m=max(r['maximum_penetration_m'] for r in tail),
        saturation_events=sum(any(r['actuator_saturated']) for r in rows),maximum_saturation_fraction=max(max(r['actuator_saturation_fraction']) for r in rows),
        fixed_support_maximum_position_error_m=max(r['sphere_displacement_from_anchor_m'] for r in rows),
        qpos_change_rad=(np.asarray(rows[-1]['qpos'])[:25]-np.asarray(protocol['qpos'])[:25]).tolist(),
        modeled_normal_load_n=source['best']['actual_friction']['total_normal_force_n'],
        realized_last_normal_load_n=sum(c['normal_force_n'] for c in rows[-1]['contacts']),
        modeled_force_residual_n=source['best']['actual_friction']['force_residual_n'],modeled_torque_residual_nm=source['best']['actual_friction']['torque_residual_nm'],
        learned_cause='Actuator-coordinate preload sustains some isolated contacts, but distal calibration does not transfer to proximal receiver load. Simultaneous commands change the realized geometry, add contacts and overcompress the sphere; the weld balances a large residual wrench. Force saturation is not the cause. Tendon sum control is correct, but does not independently hold distal joint split. Little isolated contact is not sustained in this frozen sweep.',
        no_second_object=True,no_handoff=True,no_rl=True,no_skin=True,no_shape_retest=True,
        archived_workspace_fractions=dict(thumb=.9559782183972225,index=1.,opposition=.9665998246424643),
        figures=[f'docs/figures/phase3C12B/{n}.pdf' for n in FIGURES])
    return b.save('phase3c12b_summary.json',summary),rows


def reports(summary,rows):
    audit=b.read('actuation_audit.json'); protocol=b.read('receiver_protocol.json'); source=b.source_receiver(); release=b.read('release_results.json')
    selected=summary['selected_primitives']; labels=('little','palm','middle','ring'); bysurface={r['surface']:r for r in selected}
    validation=b.read('validation.json') if (b.OUTPUT/'validation.json').exists() else {'pytest':'NOT RUN YET','diff_check':'NOT RUN YET'}
    videos=b.read('videos.json') if (b.OUTPUT/'videos.json').exists() else {'generated':[]}
    def primitive_values(key): return '; '.join(s+' '+str(bysurface[s][key]) for s in labels)
    answers=[BRANCH,BASE,
      'All 21 are position-like compiled general actuators: 17 joint transmissions, four fixed-tendon transmissions.',
      'MF/RF: J4 and J3 joint actuators, J0 -> J2+J1 tendon. LF additionally has J5. FF similarly J4/J3/J0. Thumb J5..J1 are joint actuators. Wrist and forearm are joint actuators.',
      'ctrl is target actuator length: joint angle for unit-gear hinges, sum of J2+J1 angles for fixed tendons. Fixed-tendon units here are generalized angular length, not metres and not normalized input.',
      'MF/RF/LF direct joints kp=1; J0 tendon kp=0.5. Wrist WRJ2=10, WRJ1=8; forearm=10. Thumb J5/J4/J3/J2/J1=0.4/1/0.5/1.5/1. Actuator velocity damping=0; native passive joint damping remains.',
      'MRL J4 [-0.349066,0.349066]; J3 [-0.261799,1.5708]; J0 [0,3.1415]; LFJ5 [0,0.785398]. Full wrist/thumb/index ranges in actuation_audit.json.',
      'MRL and FF actuators [-1,1] in actuator force coordinates (joint torque for hinges); WRJ2 and forearm [-10,10], WRJ1 [-5,5]. Thumb [-3,3], [-2,2], then three [-1,1].',
      'Four fixed tendons, unit coefficients: FFJ2+FFJ1, MFJ2+MFJ1, RFJ2+RFJ1, LFJ2+LFJ1.',
      'No equality fixing the J2/J1 split. The shared actuator applies the same generalized torque contribution to both; 21 controls do not independently command 25 hand DOFs.',
      'Pose-to-ctrl sum mapping was correct. Interpreting a kinematically assigned pose as actuator-held independent joints, or tangent targets as persistent preload, was not justified.',
      f"No measured actuator-force saturation in the primary construction: {summary['saturation_events']} timesteps; peak fraction {summary['maximum_saturation_fraction']:.9g}. Some requested primitive commands clipped at ctrl bounds; that is not force saturation.",
      'Little, palm/root (via wrist/forearm), middle, ring. Thumb/index audited, not dynamically swept.',
      '[0,0.01,0.025,0.05,0.10,0.20,0.30,0.40] times the precomputed unit-max actuator-coordinate normal direction. Exact signed vectors and clipped targets are stored. Eight values x four surfaces x 500 steps. Items 15-22 describe the selected examples: little=0.40, palm=0.05, middle=0.40, ring=0.05. Actuator vector orders are LFJ5/J4/J3/J0, WRJ2/WRJ1/forearm_PS, MFJ4/J3/J0, RFJ4/J3/J0 respectively; values are final-100 means unless stated otherwise.',
      primitive_values('mean_target_error'),primitive_values('mean_actuator_force'),primitive_values('mean_force_n'),primitive_values('force_to_weight'),
      'Longest positive intended-contact run (steps): '+primitive_values('max_contact_run_steps')+'. Final 100-step persistence: '+primitive_values('persistent_final_100')+'.',
      'No geom switching in isolated curves; disappearance is logged separately. Receiver adds ring-proximal and middle/little middle-link contacts.',
      'First-to-last observed contact tangential migration (m): '+primitive_values('tangential_contact_migration_m')+'. Lost contact is not interpolated to the end.',
      'Final-100 force variance (N^2): '+primitive_values('variance_force_n2')+'. Full curves and ranges are stored; no publication stability cutoff.',
      'Not for all M/R/L: middle and ring demonstrate sustained non-saturated preload; little does not in the frozen isolated sweep. Palm also sustains preload. This is not proof that little cannot preload at another local contact.',
      'Exact source qpos and sphere pose: fixed_support/receiver_protocol.json. ROLE_MRL_05 palm center '+vec(source['network']['center_palm_m'])+' m; no new basin search.',
      vec(protocol['orientation_rad'])+' rad in order forearm_PS, WRJ1, WRJ2. Reused C12A best; no outcome-based retuning.',
      str(protocol['ramp_steps'])+' steps (0.4 s), all M/R/L concurrently.',str(protocol['settle_steps'])+' further steps (1.0 s).',
      '; '.join('/'.join(c['geom_names']) for c in summary['receiver_last_contacts']),
      '; '.join(c['surface']+' '+vec(c['normal_palm']) for c in summary['receiver_last_contacts'])+' (palm frame, final welded sample).',
      '; '.join(c['surface']+' '+str(c['normal_force_n'])+' N' for c in summary['receiver_last_contacts']),
      '; '.join(c['surface']+' '+str(c['tangential_force_n'])+' N' for c in summary['receiver_last_contacts']),
      '; '.join(c['surface']+' '+str(c['rho_translation']) for c in summary['receiver_last_contacts'])+'; spin/rolling moments separately logged, not mixed into rho.',
      vec(summary['mean_hand_force_world_n'])+' N (last-100 mean world vector).',vec(summary['mean_hand_torque_world_nm'])+' N*m (about sphere COM).',
      vec(summary['mean_weld_force_world_n'])+' N.',vec(summary['mean_weld_torque_world_nm'])+' N*m.',
      vec(summary['mean_free_net_force_world_n'])+f" N; norm {summary['mean_free_net_force_norm_n']:.9g} N, {summary['mean_free_net_force_weight_ratio']:.9g} weights.",
      vec(summary['mean_free_net_torque_world_nm'])+f" N*m; norm {summary['mean_free_net_torque_norm_nm']:.9g}.",
      f"{summary['saturation_events']} force-saturated steps; peak actuator force/limit {summary['maximum_saturation_fraction']:.9g}.",
      f"Maximum {summary['maximum_penetration_m']:.9g} m; final-100 maximum {summary['tail_maximum_penetration_m']:.9g} m, exceeding inherited 0.003-m reference. Fixed-support position error peaks at {summary['fixed_support_maximum_position_error_m']:.9g} m (also just above frozen 1e-6-m numerical target).",
      'Persistent M/R/L contacts form, but NOT an admissible stable receiver: excessive overlap, large unsupported wrench and force variation. Do not count weld support as success.',
      str(protocol['release_step'])+' was frozen; release cancelled at the construction gate.'
    ]
    answers += ['NOT EXECUTED / N/A: weld remained active.']*9
    answers += ['N/A: no release.']*7
    answers += ['RECEIVER_CONSTRUCTION_FAILURE, with GROSS_PENETRATION observed during construction. No post-release failure label is assigned.',
      'Two predicted contacts: middle proximal + little proximal; normal-only total load 0.13431159757705 N.',
      'Five final contacts across M/R/L: middle proximal+middle link, ring proximal, little proximal+middle link.',
      '; '.join(c['surface']+' '+str(c['normal_difference_deg'])+' deg' for c in summary['receiver_last_contacts'] if c['predicted_pair'])+'; other contacts have no predicted pair.',
      f"Predicted {summary['modeled_normal_load_n']:.12g} N vs final realized sum {summary['realized_last_normal_load_n']:.12g} N; distinct realized geometry.",
      'Predicted normal-only rho=0; realized per-contact values in item 32. Lower force utilization alone is not equilibrium.',
      'Predicted '+vec(summary['modeled_force_residual_n'])+' N vs realized last-100 mean '+vec(summary['mean_free_net_force_world_n'])+' N.',
      'Predicted '+vec(summary['modeled_torque_residual_nm'])+' N*m vs realized last-100 mean '+vec(summary['mean_free_net_torque_world_nm'])+' N*m.',
      summary['learned_cause']+f" The final realized normal cone still contains the gravity-support direction (angular residual {summary['realized_normal_cone']['angular_distance_deg']:.9g} deg). This is force-cone feasibility only, not realized load allocation or full-wrench equilibrium.",
      'Contact realization/load allocation remains the blocker. Correct ctrl semantics and increased virtual target establish some primitives but do not realize the assumed equilibrium network.',
      'Persistent M/R/L contacts occur simultaneously, but not with acceptable overlap and balanced load. Little isolated preload remains unestablished.',
      'It forms contacts, not a valid receiver network.', 'Not tested: gate correctly prevented release.', 'No validated 1000-step receiver.',
      'None. Exact failed construction and pre-release integration state are preserved, not labeled validated.',
      'Virtual offsets calibrated on isolated distal contacts were not transferable to the proximal multi-contact network. Additional contacts, large overlap and unbalanced wrench emerged without force saturation. Initial soft-weld compliance was a separate setup defect, now exposed and documented.',
      'No.', 'No.', 'No.', 'Yes.', 'Yes.', 'Yes.',
      'Bounded Phase 3C-1.2B follow-up: contact-specific proximal MRL preload/load-allocation debugging at the same frozen orientation and basin, with fixed-sphere support validated first. Do not resume handoff, shape trials, skin, RL or object B.',
      validation['pytest'],validation['diff_check'],f'20 PDFs under docs/figures/phase3C12B/. Four post-release plots explicitly say NOT EXECUTED.',
      str(videos['generated'])+'; no release/success/failure-of-release video exists because no release executed.',
      'docs/PHASE3C12B_RESULTS.md; docs/PHASE3C12B_ACTUATION_AUDIT.md; outputs/phase3C12B/fixed_support/phase3c12b_summary.json, fixed_sphere_primitives.json, receiver_protocol.json, receiver_construction.json, release_results.json, *.npz and primitives/*.npz. Earlier soft-weld setup diagnostics remain under outputs/phase3C12B/.'
    ]
    assert len(answers)==86, len(answers)
    summary['answers']={str(i):v for i,v in enumerate(answers,1)}; summary['videos']=videos['generated']
    summary['source_C12A_hashes']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in b.a.OUTPUT.glob('*.json')}
    b.save('phase3c12b_summary.json',summary)
    sections={1:'Actuation audit',13:'Fixed-sphere preload',24:'Welded receiver construction',42:'Weld release',60:'Theory vs reality',68:'Final decision'}
    lines=['# Phase 3C-1.2B final report','',
       '**RECEIVER_CONSTRUCTION_FAILURE. Zero weld releases. No validated receiver.**','',
       'Two namespaces distinguish a setup diagnostic using the inherited soft weld from the fixed-support implementation. The soft support moved millimetres and did not satisfy the requested premise; all its evidence is preserved. The new temporary external-support weld uses solref=[0.004,1], solimp=[0.9999,0.9999,0.001,0.5,2]. This is an explicitly disclosed support-constraint implementation change, not a hand/contact-physics or actuator-gain change. Sphere translation is numerically constrained, not mathematically exact. No friction, geometry, solver, timestep, native coupling, joint bounds or gains changed.','']
    for i,value in enumerate(answers,1):
        if i in sections: lines+=['## '+sections[i],'']
        lines+=[f'{i}. {value}','']
    lines+=['## Scope and causal limitations','',
        'The observed network is not the predicted fixed network. In particular, distal-only primitive calibration does not establish proximal-contact force gains in ROLE_MRL_05. Middle/little virtual offset choices were frozen before each construction; little used an explicitly uncalibrated maximum-sweep diagnostic fallback. No controller was retuned using release outcomes. One inherited-soft-weld setup construction and one fixed-support construction were executed; neither released. No broad search or failure batch occurred.','',
        'The receiver readiness gate uses necessary multi-contact persistence, actual force-limit saturation, environment support, the inherited 0.003-m penetration reference and a 1-micrometre fixture position target. It does not silently decide a publication threshold for force variance or acceptable free wrench. Raw wrench/variance values are reported. In this experiment gross overlap independently blocks release; the fixture peak error also marginally exceeds its declared target. This phase does not prove impossibility of MRL storage or justify morphology/skin conclusions.','',
        'The frozen 0.003-m overlap reference is inherited from Phase 3C, not newly approved as a publication success criterion. Likewise, the 1e-9-N positive-force test is numerical detection, not weight-bearing adequacy. No post-release retention percentages or peak speeds are invented from the welded trajectory.','',
        'Theoretical vs realized comparisons use exact geom pair identity. Angular differences are measured in the actual palm frame; added contacts are not silently paired to old contacts. Weld wrench is recovered from the equality-only generalized force J^T f and mapped through the sphere COM Jacobian. Contact force and torque include solver-reported spin/rolling moments. Counterfactual acceleration is a frozen-contact instantaneous wrench prediction, not a simulated free trajectory.','',
        'Official reference: [MuJoCo actuation and constraint force mapping](https://mujoco.readthedocs.io/en/stable/computation/index.html). The compiled gain/bias arrays determine ctrl semantics; a fixed tendon controls a joint-angle sum and not the independent split.','',
        '## Unresolved PI decision','',
        'configs/phase3C12B_weld_release_receiver.yaml: publication thresholds for force stability, receiver wrench and overlap remain undecided. No such threshold was silently supplied.','',
        '## Reproduction','',
        'Use only .\\.venv\\Scripts\\python.exe. scripts/run_phase3c12b.py exposes explicit audit, primitives, construction and release stages; existing outcome files prevent accidental reruns. scripts/analyze_phase3c12b.py consumes artifacts only. scripts/generate_phase3c12b_videos.py renders saved qpos without dynamics. Preserve both output namespaces.','']
    (ROOT/'docs/PHASE3C12B_RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')
    lines=['# Phase 3C-1.2B compiled actuation audit','',
      'Compiled fields, not actuator names alone, establish position-servo semantics. All gains are fixed, bias affine, activation dynamics absent, gear=1. Scalar actuator force is clipped kp*(ctrl-actuator_length). Actuator velocity bias is zero. Native passive damping is 0.05 for finger/forearm DOFs and 0.5 for wrist DOFs. Joint limits and actuator bounds are not changed.','',
      '| Actuator | Transmission target | kp | ctrl range | force range |','|---|---|---:|---|---|']
    for r in audit['actuators']: lines.append(f"| {r['name']} | {r['transmission']}: {r['target']} | {r['effective_kp']} | {r['ctrlrange']} | {r['forcerange']} |")
    lines+=['','## Coupling','',audit['distal_coupling'],'',
        'All four fixed tendons have two unit joint terms. There is no equality tying J1 to J2. Passive dynamics, limits and contact redistribute the tendon-controlled total between them. The current model has one equality: the temporary sphere weld. Its presence does not indicate a digit coupling.','',
        '## Complete compiled parameters','',
        'The machine audit includes every gear component, ctrllimited/forcelimited flag, gainprm/biasprm/dynprm array, transmission enum and target, joint range/stiffness/damping, tendon springlength/stiffness/damping, and native-hand/contact fingerprint: outputs/phase3C12B/fixed_support/actuation_audit.json.','',
        'All 500 debug samples per primitive and 700 construction samples record ctrl, actuator force/limit fraction, actual saturation, joint qpos/qvel, tendon lengths/velocities and target error. Ctrl clipping and force saturation are independent measurements.','',
        'The original tangent-position command mapping was dimensionally correct but did not create an intentional persistent servo error. The virtual target now creates such error in actuator coordinates. Its Jacobian-based direction is a local engineering approximation; it cannot guarantee the passive distal split or an unchanged contact network.','',
        'Implementation: seqgrasp/phase3c12b.py: actuation_audit, transmission_matrix, normal_virtual_direction, saturation and record. Official [MuJoCo actuation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html).']
    (ROOT/'docs/PHASE3C12B_ACTUATION_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def figures(summary,rows):
    FIG.mkdir(parents=True,exist_ok=True); plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,index,title,note):
        fig.suptitle(title,fontsize=16,x=.07,ha='left',y=.98); fig.text(.07,.025,note,fontsize=9,color='#475569')
        fig.subplots_adjust(left=.10,right=.95,bottom=.19,top=.84,hspace=.55,wspace=.35); fig.savefig(FIG/(FIGURES[index]+'.pdf')); plt.close(fig)
    def small(): return plt.subplots(figsize=(11,6.2))
    def vectors(index,field,title,unit,note):
        fig,ax=small(); v=np.asarray([r[field] for r in rows]); x=np.arange(1,len(rows)+1)
        for i,n in enumerate('XYZ'): ax.plot(x,v[:,i],label=n)
        ax.plot(x,np.linalg.norm(v,axis=1),'k--',label='Norm'); ax.axvline(200,color='#94a3b8',ls=':'); ax.legend(); ax.set(xlabel='Welded construction step',ylabel=unit)
        save(fig,index,title,note)
    curves=summary['primitive_curves']; selected={r['surface']:r for r in summary['selected_primitives']}; source=b.source_receiver()
    fig,ax=small(); ax.axis('off')
    texts=[('21 position servos -> 25 hand DOFs','17 joint actuators + four fixed-tendon actuators'),('MRL / FF distal actuation','J0 controls J2 + J1, with unit coefficients; the split is not fixed.'),('Scalar servo law','Force = clipped kp x (ctrl - actuator length). No actuator velocity damping.'),('Persistent preload','Blocked virtual target -> nonzero error -> actuator force -> contact wrench.')]
    for i,(head,body) in enumerate(texts): ax.text(.01,.91-i*.24,head,weight='bold',fontsize=12); ax.text(.01,.83-i*.24,body,fontsize=11)
    save(fig,0,'Shadow Hand actuation: do not equate joints with controls','Native hand/contact parameters unchanged. Complete compiled arrays in actuation_audit.json.')
    sample=b.load_series(selected['middle']['timeseries']); ids=selected['middle']['actuator_ids']; x=np.arange(1,501)
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    for i in ids:
        axes[0].plot(x,[r['ctrl'][i] for r in sample],label=f'ctrl {i}'); axes[0].plot(x,[r['actuator_length'][i] for r in sample],ls='--',label=f'length {i}')
        axes[1].plot(x,[r['actuator_force'][i] for r in sample],label=f'actuator {i}')
    axes[0].set(ylabel='Actuator coordinate (rad-like)',title='Middle: virtual offset 0.40'); axes[0].legend(ncol=3,fontsize=8)
    axes[1].set(xlabel='Fixed-support step',ylabel='Actuator force'); axes[1].legend(fontsize=8,ncol=3)
    save(fig,1,'Command, transmission response, and measured servo force','Solid: command; dashed: actual actuator length. Tendon length is a coupled angle sum.')
    sample=b.load_series(selected['little']['timeseries']); fig,ax=small()
    ax.plot(x,[surface_force(r,'little') for r in sample],label='Little normal force'); ax.axhline(summary['weight_n'],color='black',ls='--',label='Weight reference'); ax.legend(); ax.set(xlabel='Fixed-support step',ylabel='Normal force (N)')
    save(fig,2,'Little isolated preload: transient contact, then loss','Maximum frozen virtual offset 0.40. Final-100-step force is zero; no sustained little primitive.')
    fig,axes=plt.subplots(1,3,figsize=(12,6.2))
    for ax,s in zip(axes,('middle','ring','little')):
        group=[r for r in curves if r['surface']==s]; ax.errorbar([r['virtual_offset'] for r in group],[r['mean_force_n'] for r in group],yerr=[np.sqrt(r['variance_force_n2']) for r in group],fmt='o-'); ax.set(title=s,xlabel='Virtual actuator offset',ylabel='Tail normal force (N)')
    save(fig,3,'MRL force curves: final 100 of 500 steps','Error bars: one temporal standard deviation, not confidence intervals. Same frozen eight offsets.')
    fig,ax=small(); ax.axis('off')
    for i,(head,body) in enumerate([('Independent variable','Virtual command offset in actual actuator coordinates, not penetration.'),('Intended mechanism','Contact prevents reaching target; persistent error generates contact force.'),('Observed limitation','Tendon coupling, bound clipping and contact migration change the response.'),('Calibration transfer','Distal single-contact calibration does not specify proximal receiver force.')]):
        ax.text(.01,.90-i*.24,head,fontsize=12,weight='bold'); ax.text(.01,.82-i*.24,body,fontsize=11)
    save(fig,4,'Virtual targets create force, but not a prescribed fingertip wrench','No F = kp x offset shortcut was used to equate actuator force with sphere normal force.')
    fig,ax=small(); ax.plot(np.arange(1,len(rows)+1),[max(r['actuator_saturation_fraction']) for r in rows]); ax.axhline(1,color='#dc2626',ls='--'); ax.set(xlabel='Construction step',ylabel='Maximum actuator force / limit',ylim=(0,1.1))
    save(fig,5,'Measured force saturation: no construction timestep saturated','Ctrl clipping is logged independently. The force-limit line is the actual compiled limit.')
    fig,axes=plt.subplots(1,2,figsize=(11,6.2)); xw=np.arange(1,len(rows)+1)
    axes[0].plot(xw,[r['sphere_displacement_from_anchor_m']*1e6 for r in rows]); axes[0].axhline(1,color='red',ls='--'); axes[0].set(xlabel='Construction step',ylabel='Sphere anchor error (micrometres)')
    axes[1].plot(xw,[r['maximum_penetration_m']*1000 for r in rows]); axes[1].axhline(3,color='red',ls='--'); axes[1].set(xlabel='Construction step',ylabel='Maximum overlap (mm)')
    save(fig,6,'Welded MRL construction: fixed support does not imply valid contact','One primary construction. Red lines: declared fixture numerical target and inherited overlap reference.')
    fig,ax=small()
    for s in ('middle','ring','little'): ax.plot(xw,[surface_force(r,s) for r in rows],label=s)
    ax.axvline(200,color='#64748b',ls='--',label='Ramp ends'); ax.legend(); ax.set(xlabel='Welded construction step',ylabel='Normal force (N)')
    save(fig,7,'Simultaneous MRL ramp forms persistent but excessive loads','Commands ramp together over 200 steps, then hold for 500. No release at step 700.')
    fig,ax=small(); support=np.array([[s in r['topology'] for r in rows] for s in ('middle','ring','little','palm')]); ax.imshow(support,aspect='auto',origin='lower',extent=[1,len(rows),-.5,3.5],vmin=0,vmax=1,cmap='Blues',interpolation='nearest'); ax.set(yticks=range(4),yticklabels=['middle','ring','little','palm'],xlabel='Welded construction step')
    save(fig,8,'Contact persistence is not receiver validation','Blue: positive normal contact. Geometry, overlap and unsupported wrench must also be considered.')
    vectors(9,'free_net_force_world_n','Counterfactual free-sphere net force before release','Force (N)','Hand contacts + gravity only; weld excluded. This is not a post-release trajectory.')
    vectors(10,'free_net_torque_world_nm','Counterfactual free-sphere net torque before release','Torque (N m)','Moment about sphere COM includes contact spin/rolling terms. Weld excluded.')
    fig,ax=small()
    for field,label in [('weld_force_world_n','Weld reaction norm'),('hand_force_world_n','Hand contact resultant norm')]: ax.plot(xw,[norm(r[field]) for r in rows],label=label)
    ax.axhline(summary['weight_n'],color='black',ls='--',label='Sphere weight'); ax.legend(); ax.set(xlabel='Welded construction step',ylabel='Force magnitude (N)')
    save(fig,11,'The weld carries the unbalanced receiver wrench','A large weld reaction is evidence against independent support, not receiver success.')
    for i in range(12,16):
        fig,ax=small(); ax.axis('off'); ax.text(.5,.63,'NOT EXECUTED',ha='center',fontsize=26,color='#b91c1c',weight='bold'); ax.text(.5,.43,'Weld release was blocked by receiver construction failure.',ha='center',fontsize=13); ax.text(.5,.26,'No free-sphere motion, contact-force or net-wrench trace exists.',ha='center',fontsize=11)
        save(fig,i,FIGURES[i].replace('_',' ').capitalize(),'N/A is not zero retention. Do not infer post-release success or failure from a welded run.')
    fig,ax=small(); modeled=[.10425577113835,0,.0300558264387]; realized=[sum(c['normal_force_n'] for c in summary['receiver_last_contacts'] if c['surface']==s) for s in ('middle','ring','little')]; xs=np.arange(3)
    ax.bar(xs-.18,modeled,.36,label='Modeled normal-only load'); ax.bar(xs+.18,realized,.36,label='Realized last sample'); ax.set(xticks=xs,xticklabels=['middle','ring','little'],ylabel='Normal force (N)'); ax.legend()
    save(fig,16,'Modeled two-contact network becomes five realized contacts','Final sample includes extra middle-link contacts and ring contact. Loads are not calibrated equivalents.')
    fig,ax=small(); matched=[c for c in summary['receiver_last_contacts'] if c['predicted_pair']]; ax.bar([c['surface'] for c in matched],[c['normal_difference_deg'] for c in matched]); ax.set(ylabel='Normal change in palm frame (deg)')
    save(fig,17,'The original contact normals do not survive unchanged','Exact geom-pair matches only. Three added contacts have no modeled counterpart.')
    fig,ax=small(); cs=summary['receiver_last_contacts']; ax.bar([f"{c['surface']} {i+1}" for i,c in enumerate(cs)],[c['rho_translation'] for c in cs]); ax.axhline(1,color='red',ls='--'); ax.set(ylabel='Translational friction utilization',ylim=(0,1.1))
    save(fig,18,'Realized friction use differs from the theoretical rho = 0 solution','Utilization excludes spin/rolling. Feasible individual friction ratios do not establish net equilibrium.')
    fig,ax=small(); ax.axis('off')
    texts=[('Actuation mapping verified','Persistent isolated middle/ring/palm force is possible; little sweep loses contact.'),('Receiver geometry changed','Five contacts replace the modeled two; overlap reaches millimetres.'),('Weld carries the residual','Mean unsupported force is about 33 sphere weights; no actuator saturation.'),('Correct stopping point','RECEIVER_CONSTRUCTION_FAILURE. Zero releases; no validated receiver.')]
    for i,(head,body) in enumerate(texts): ax.text(.01,.91-i*.24,head,weight='bold',fontsize=12,color='#b91c1c' if i==3 else '#1e3a8a'); ax.text(.01,.83-i*.24,body,fontsize=11)
    save(fig,19,'Phase 3C-1.2B causal summary','No handoff, shape retest, skin, RL, object B or new basin search. Leave uncommitted for PI review.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--reports-only',action='store_true'); args=parser.parse_args()
    summary,rows=summarize(); reports(summary,rows)
    if not args.reports_only: figures(summary,rows)
    print('Wrote Phase 3C-1.2B evidence report'+('' if args.reports_only else ' and 20 PDFs'))
