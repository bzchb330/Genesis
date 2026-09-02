"""Offline P0 comparison, PI report and figures. Never runs dynamics."""
from __future__ import annotations
import argparse
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from seqgrasp import phase3cp0 as p, contact_bench as b

FIG=p.ROOT/'docs/figures/phase3CP0'
FIGURES=['contact_bench_geometry','current_contact_parameters','normal_load_vs_penetration_current_physics',
 'dimensionless_deformation_vs_load','task_force_scale_reference','contact_force_monotonicity',
 'load_unload_hysteresis','contact_energy_sanity','timestep_sensitivity','solver_sensitivity',
 'effective_contact_stiffness','candidate_contact_parameter_comparison','candidate_force_deformation_curves',
 'candidate_timestep_robustness','selected_physics_validation','phase3CP0_physical_admissibility_summary']


def read_suite(name): return json.loads((p.OUTPUT/name/'results.json').read_text())
def first_rows(suite): return [r for r in suite['rows'] if r['tag'].endswith('repeat_0')]
def trace(row): return b.load_trace(p.ROOT/row['trace'])


def summarize_suite(suite):
    rows=first_rows(suite); delta=np.array([r['mean']['signed_overlap_m'] for r in rows]); force=np.array([r['mean']['normal_force_n'] for r in rows])
    grad=np.gradient(force,delta); curves=[]; repeat_error=0.
    for i,r in enumerate(rows):
        curves.append(dict(load_n=r['load_n'],overlap_m=r['mean']['overlap_m'],delta_over_radius=r['mean']['delta_over_radius'],
                           normal_force_n=r['mean']['normal_force_n'],normal_force_variance_n2=r['variance']['normal_force_n'],
                           overlap_variance_m2=r['variance']['overlap_m'],settling_after_ramp_seconds=r['settling_after_ramp_seconds'],
                           secant_stiffness_npm=float(force[i]/delta[i]),gradient_stiffness_npm=float(grad[i]),
                           residual_kinetic_energy_j=r['mean']['kinetic_energy_j']))
        other=next(x for x in suite['rows'] if x['tag']==r['tag'].replace('repeat_0','repeat_1'))
        repeat_error=max(repeat_error,float(np.max(abs(trace(r)['samples']-trace(other)['samples']))))
    sensitivity=[]
    for load in p.config()['bench']['sensitivity_loads_n']:
        primary=next(r for r in rows if r['load_n']==load)
        cases=[primary]+[r for r in suite['rows'] if r['load_n']==load and (r['tag'].startswith('dt_') or r['solver_diagnostic'])]
        for r in cases:
            ref=primary['mean']['overlap_m']; change=r['mean']['overlap_m']-ref
            sensitivity.append(dict(load_n=load,timestep_s=r['timestep_s'],solver_diagnostic=r['solver_diagnostic'],
                                    penetration_m=r['mean']['overlap_m'],delta_difference_m=change,relative_delta_difference=change/ref,
                                    normal_force_n=r['mean']['normal_force_n'],force_difference_n=r['mean']['normal_force_n']-primary['mean']['normal_force_n'],
                                    force_std_n=float(np.sqrt(r['variance']['normal_force_n'])),
                                    settling_after_ramp_seconds=r['settling_after_ramp_seconds'],
                                    work_balance_residual_j=r['final_work_balance_residual_j']))
    dt=[r for r in sensitivity if not r['solver_diagnostic']]; solver=[r for r in sensitivity if r['solver_diagnostic']]
    cycles=[{k:r[k] for k in ('load_n','final_contact_work_j','final_external_work_j','final_gravity_work_j',
                              'final_work_balance_residual_j','final_kinetic_energy_j','unload_final_signed_overlap_m','contact_loss_transitions')}
            for r in suite['rows'] if r['cycle']]
    return dict(name=suite['name'],candidate=suite['candidate'],trials=suite['trials'],curves=curves,
                monotonic=bool(np.all(np.diff(delta)>0) and np.all(np.diff(force)>0)),repeat_maximum_absolute_trace_difference=repeat_error,
                maximum_loaded_contact_loss_count=max(r['contact_loss_transitions'] for r in suite['rows'] if not r['cycle']),
                negative_normal_force_samples=sum(r['negative_normal_force_samples'] for r in suite['rows']),
                maximum_held_force_step_n=max(r['maximum_hold_force_step_n'] for r in rows),
                maximum_tail_force_variance_n2=max(r['variance']['normal_force_n'] for r in rows),
                maximum_tail_kinetic_energy_j=max(r['mean']['kinetic_energy_j'] for r in rows),
                maximum_dt_relative_delta_difference=max(abs(r['relative_delta_difference']) for r in dt),
                maximum_solver_relative_delta_difference=max(abs(r['relative_delta_difference']) for r in solver),
                sensitivity=sensitivity,cycles=cycles,
                contact_work_all_cycles_nonpositive=all(r['final_contact_work_j']<=0 for r in cycles),
                current_setting_settling_range_s=[min(r['settling_after_ramp_seconds'] for r in rows),max(r['settling_after_ramp_seconds'] for r in rows)])


def summarize():
    cfg=p.frozen_config(); names=[cfg['legacy_name']]+[o['name'] for o in cfg['candidate_options']]
    suites=[summarize_suite(read_suite(n)) for n in names]
    summary=dict(branch=p.BRANCH,base_commit=p.BASE,classification=p.read('legacy_classification.json'),suites=suites,
                  total_bench_trials=sum(s['trials'] for s in suites),selected_candidate=None,approved_physics_v1=None,
                  hand_primitive=dict(executed=False,reason='CP-C legacy; candidates require PI approval; material elastic constants unspecified.',surfaces_tested=[],
                                      planned_surfaces=cfg['hand_primitive']['surfaces'],planned_forces_n=cfg['hand_primitive']['desired_forces_n']),
                  regression=p.read('legacy_regression.json'),scope=p.scope(),
                  figures=[f'docs/figures/phase3CP0/{x}.pdf' for x in FIGURES],
                  frozen_kinematic_resources=dict(thumb=.9559782183972225,index=1.,thumb_index_opposition=.9665998246424643),
                  static_network_note='Conditional idealized ROLE_MRL_05 feasibility preserved; not a realized receiver.',
                  videos=p.read('videos.json') if (p.OUTPUT/'videos.json').exists() else {})
    p.save('physics_registry.json',dict(active_production=cfg['legacy_name'],legacy_snapshot='current_physics.json',
           options=cfg['candidate_options'],all_options_pi_approved=False,PHYSICS_V1_RIGID_CONTACT=None,
           scope='Isolated normal sphere-plane response, 0.01-1 N, 0.001-0.004 s. No material calibration, tangential validation or hand control validation.'))
    return p.save('phase3cp0_summary.json',summary)


def reports(s):
    current=p.read('current_physics.json'); cfg=p.config(); legacy=s['suites'][0]; curves=legacy['curves']
    validation=p.read('validation.json') if (p.OUTPUT/'validation.json').exists() else dict(pytest='NOT RUN YET',diff_check='NOT RUN YET')
    vec=lambda key: ', '.join(f"{r[key]:.9g}" for r in curves)
    options='; '.join(f"{o['name']}: solref={o['solref']}, solimp={o['solimp']}" for o in cfg['candidate_options'])
    answers=[p.BRANCH,p.BASE,'0.002 s.','Newton, 100 iterations, tolerance 1e-8; Euler; elliptic cone, impratio=10.',
      'Sphere condim=6, hand condim=3; actual sphere-hand/bench contacts dim=6 because sphere priority=1.',
      'Sphere [0.5,0.01,0.003]; hand [1,0.005,0.0001]; runtime pair [0.5,0.5,0.01,0.003,0.003]. Unchanged.',
      'Sphere/runtime [0.02,1]; hand [0.005,1].','Sphere/runtime [0.9,0.95,0.001,0.5,2]; hand [0.5,0.99,0.0001,0.5,2].',
      '0 m margin and gap.','0.0125 m.',str(current['sphere_mass_kg'])+' kg.',str(current['sphere_weight_n'])+' N.',
      'No: MATERIAL ELASTIC CONSTANTS NOT SPECIFIED BY PI. 459 baseline tracked text files searched; no justified E/nu or restitution found.',
      'One free 25-mm sphere, horizontal infinite fixed plane, no hand/tendon/servo/weld. Exact compiled mass/inertia and contact-pair parameters. Starts tangent without overlap. Gravity retained; applied COM force z=mg-F_target.',
      str(cfg['bench']['loads_n'])+' N; two repeats each. 0.4-s ramp + 4-s hold. Items 16-20 follow this ascending load order.',
      vec('overlap_m')+' m.',vec('delta_over_radius')+'.',vec('normal_force_n')+' N.',vec('normal_force_variance_n2')+' N^2.',
      vec('settling_after_ramp_seconds')+' s after ramp. Engineering-only 1% force/position band and 1e-5-m/s speed bound; not publication success.',
      str(legacy['monotonic'])+': steady normal force and compression increase monotonically.',
      f"Exact repeated saved trajectories; maximum absolute difference {legacy['repeat_maximum_absolute_trace_difference']:.9g}.",
      'All four unload cycles dissipate net contact work. No residual overlap: sphere separates at zero total load and continues small free drift because external force cancels gravity. This is not positive-load contact loss.',
      f"Loaded final-200 mean kinetic energy <= {legacy['maximum_tail_kinetic_energy_j']:.9g} J. Cycle contact work is negative. External/gravity/contact work and integration remainder are logged; no exact continuum energy or global passivity proof is claimed.",
      f"dt=0.001/0.002/0.004 s. Max relative steady-overlap change {legacy['maximum_dt_relative_delta_difference']:.9g}; complete transient/settling/energy differences stored.",
      f"400 iterations / 1e-12 vs 100 / 1e-8: max relative steady-overlap change {legacy['maximum_solver_relative_delta_difference']:.9g}.",
      'No. Steady force-deformation behavior is contact-parameter dominated in this isolated geometry.',
      'CP-C for rigid-contact interpretation across the complete tested range; numerically stable, not CP-D.',s['classification']['reason'],
      'Symbolic Hertz reference only; no numerical material reference fitted.',
      'No E, nu, restitution or material identity invented. Plane is kinematically rigid but solver contact is compliant.',
      'Local finite-difference dF/d(delta): '+vec('gradient_stiffness_npm')+' N/m; secant stiffness also stored.',
      'Not over the entire requested force range: upper-load deformation is not small compared with radius. This does not claim all low-load uses fail.',
      'Yes: independent contact-parameter study required by CP-C.',options,
      '; '.join(x['name']+f": delta(weight)={x['curves'][3]['overlap_m']*1000:.9g} mm, delta(0.1343N)={x['curves'][5]['overlap_m']*1000:.9g} mm, delta(1N)={x['curves'][-1]['overlap_m']*1000:.9g} mm" for x in s['suites']),
      '; '.join(x['name']+f": maximum relative steady-overlap change={x['maximum_dt_relative_delta_difference']:.9g}" for x in s['suites']),
      'None selected or approved.',
      'All three options are numerical candidates only. Missing material data/scope require PI review; smallest penetration alone is not a selection rule.',
      'LEGACY_PHASE3C_CONTACT_PHYSICS remains production. PHYSICS_V1_RIGID_CONTACT has NOT been created/approved.',
      'Legacy parameters are frozen in current_physics.json; candidate option definitions in config/registry. No production settings were modified.',
      'NOT EXECUTED: hand force-control stage is blocked by protocol pending CP-A or PI-approved revised physics.',
      'Existing Shadow actuator semantics and transmissions were preserved; no hand controller modification.',
      'None dynamically tested in P0; planned middle/ring/little.',str(cfg['hand_primitive']['desired_forces_n'])+' N, predeclared but not executed.']
    answers+=['N/A: hand stage not executed.']*8
    answers+=['No validated bounded-force hand primitive is claimed.',
      'Yes, rejected using saved evidence only; no C12B dynamics rerun.',
      'Settled max 6.598903 mm, peak 8.333245 mm violate inherited 3-mm engineering reference. Last summed normal load 6.187809 N (~77 weights), mean unsupported residual 2.648620 N (~33 weights) and cancelling weld wrench are separately flagged. Do not interpret deep-contact normals/topology/rho as valid receiver mechanics.',
      'Reusable physical_admissibility.py logs penetration/radius/diameter ratios, total normal force/weight, residual force/torque, actual actuator saturation fractions, external-support wrench, kinetic energy and environment support. Gates require explicit labels; no default scientific acceptance threshold.',
      'PI-approved contact-physics interpretation/parameters, followed by genuinely bounded contact-force control.',
      'Numerical normal-contact response characterized; physical/material validation and production candidate approval remain pending.',
      'Candidate changes were needed for the calibration study, but no revised production parameters are selected.',
      'No: blocked and not executed.',
      'Not yet. Phase 3C-P1 requires approved contact physics and validated bounded-force primitives first.',
      'Yes.','Yes.','Yes.','Yes.',
      'PI review of P0 material assumptions and candidate options, then the gated P0 fixed-sphere force-control study. Only after validation consider Phase 3C-P1; do not start it automatically.',
      validation['pytest'],validation['diff_check'],
      '16 PDFs: requested 1-15 plus physical-admissibility summary. Selected-physics figure explicitly says no selection; four hand-primitive figures omitted because the hand stage did not run.',
      str(s['videos'])+'; no hand-force, receiver or grasp-success video.',
      'docs/PHASE3CP0_RESULTS.md; docs/PHASE3CP0_CURRENT_PHYSICS_AUDIT.md; docs/PHASE3CP0_BENCH_PROTOCOL.md; outputs/phase3CP0/phase3cp0_summary.json, current_physics.json, material_audit.json, frozen_protocol.json, physics_registry.json, legacy_classification.json, legacy_regression.json, preserved_phase3C12B_hashes.json, each version/results.json and per-trial JSON/NPZ traces.']
    assert len(answers)==72,len(answers)
    s['answers']={str(i):v for i,v in enumerate(answers,1)}; p.save('phase3cp0_summary.json',s)
    groups={1:'Current physics',14:'Isolated contact bench',25:'Numerical robustness',30:'Physical reference',34:'Physics calibration',42:'Force-control primitive',55:'Regression',58:'Final decision'}
    lines=['# Phase 3C-P0 final report','',
           '**CP-C across the full requested rigid-contact load range. Candidate options await PI approval. No hand or receiver experiment executed.**','']
    for i,a in enumerate(answers,1):
        if i in groups: lines+=['## '+groups[i],'']
        lines+=[f'{i}. {a}','']
    lines+=['## Measured steady load-deformation table','',
            '| Applied load N | Measured Fn N | Overlap mm | delta/R | Fn variance N^2 | Settling s | Secant N/m | Local gradient N/m |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in curves:
        lines.append(f"| {c['load_n']:.9g} | {c['normal_force_n']:.9g} | {c['overlap_m']*1000:.9g} | {c['delta_over_radius']:.9g} | {c['normal_force_variance_n2']:.3g} | {c['settling_after_ramp_seconds']:.3g} | {c['secant_stiffness_npm']:.6g} | {c['gradient_stiffness_npm']:.6g} |")
    lines+=['','## Options for PI review, not a selected physical model','',
            '| Model | Overlap at weight mm | At 0.134311598 N mm | At 1 N mm | Settling range s | Max timestep relative delta change |',
            '|---|---:|---:|---:|---|---:|']
    for x in s['suites']:
        lines.append(f"| {x['name']} | {x['curves'][3]['overlap_m']*1000:.9g} | {x['curves'][5]['overlap_m']*1000:.9g} | {x['curves'][-1]['overlap_m']*1000:.9g} | {x['current_setting_settling_range_s']} | {x['maximum_dt_relative_delta_difference']:.3g} |")
    lines+=['','## Energy and scope limitations','',
      'The zero-total-load unload endpoint is force-balanced away from contact: mg is cancelled by the external bench load. A small upward velocity is therefore not damped after contact vanishes. Separation/drift is expected in that neutral free state; it is not equilibrium-position settling. All observed cycle contact work is net dissipative, but this limited diagnostic does not prove global passivity. Contact elastic energy is not directly exposed as a continuum strain-energy state. Work-balance residuals retain finite-step quadrature and integrator error.','',
      'The normal sphere-plane response is independent of hand geometry and tendon control; that separation is the purpose and also a limit. This phase does not validate tangential friction, rolling/spinning, multi-contact geometry or receiver force allocation. The C12B state is rejected, not recycled to infer final normals, topology, friction utilization or morphology.','',
      'Frozen resource fractions were not recomputed: thumb 0.9559782183972225, index 1.0, opposition 0.9665998246424643. The idealized static-network result remains conditional, not a realized receiver.','',
      'TODO(PI): material elastic constants, physically justified deformation/rate scope, and contact-option approval. No physics version is promoted and no scientific threshold is silently resolved.','',
      '## Reproduction','',
      'Use only .\\.venv\\Scripts\\python.exe. scripts/run_phase3cp0.py has explicit audit/legacy/candidates/regression/hand-gate stages. Existing results are protected from overwrite; candidates require the saved CP-B/C/D decision. scripts/analyze_phase3cp0.py consumes saved arrays only. Do not bypass the hand-stage approval gate.','',
      'References: [MuJoCo constraint/contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters), [constraint computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#constraint-model).','']
    (p.ROOT/'docs/PHASE3CP0_RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')


def figures(summary):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    legacy=summary['suites'][0]; source=read_suite(legacy['name']); baseline=first_rows(source)
    def new(): return plt.subplots(figsize=(11,6.4))
    def save(fig,i,title,note):
        fig.suptitle(title,x=.07,ha='left',fontsize=15,y=.98)
        fig.text(.07,.025,'\n'.join(textwrap.wrap(note,140)),fontsize=9,color='#475569')
        fig.subplots_adjust(left=.30 if i==4 else (.23 if i==11 else .11),right=.95,bottom=.19,top=.84,wspace=.35,hspace=.5)
        fig.savefig(FIG/(FIGURES[i]+'.pdf')); plt.close(fig)
    force=np.array([r['normal_force_n'] for r in legacy['curves']]); delta=np.array([r['overlap_m'] for r in legacy['curves']])
    fig,ax=new(); ax.set_aspect('equal'); ax.add_patch(plt.Circle((0,.0125),.0125,color='#df941e')); ax.axhline(0,color='#475569',lw=4)
    ax.annotate('',xy=(0,.0125),xytext=(0,.039),arrowprops={'arrowstyle':'->'})
    ax.text(.004,.031,'Total imposed load:\nF_target downward',fontsize=10)
    ax.text(-.027,.005,'R = 12.5 mm'); ax.text(-.027,-.006,'Rigid plane | no hand, servo or weld'); ax.set(xlim=(-.035,.07),ylim=(-.01,.045)); ax.axis('off')
    save(fig,0,'Isolated normal-contact bench: total load is prescribed','Gravity retained. Applied COM force is mg - F_target; sphere starts tangent with zero velocity.')
    fig,ax=new(); ax.axis('off'); lines=[('Compiled runtime pair','condim 6; friction [0.5, 0.5, 0.01, 0.003, 0.003]'),('Normal-contact law','solref [0.02, 1]; solimp [0.9, 0.95, 0.001, 0.5, 2]'),('Numerical settings','Euler; Newton; 100 iterations; tolerance 1e-8; dt 0.002 s'),('Provenance','Sphere priority 1 overrides the hand default at priority 0.')]
    for i,(head,body) in enumerate(lines): ax.text(0,.92-i*.24,head,weight='bold',fontsize=12); ax.text(0,.83-i*.24,body,fontsize=11)
    save(fig,1,'Current contact parameters are not material elastic constants','Young modulus and Poisson ratio are unspecified. No quantitative Hertz calibration is claimed.')
    fig,ax=new(); ax.plot(force,delta*1000,'o-'); ax.set(xlabel='Measured steady normal force (N)',ylabel='Numerical overlap (mm)')
    save(fig,2,'Legacy contact compliance reaches millimetre-scale deformation','Final 200 of 2000 held-load steps. All ten frozen loads; two repeats reproduce exactly.')
    fig,ax=new(); ax.plot(force,delta/.0125,'o-'); ax.set(xlabel='Measured normal force (N)',ylabel='Signed deformation / sphere radius')
    save(fig,3,'Deformation relative to the 12.5-mm radius','No universal admissible delta/R cutoff is imposed. Values describe this geometry and load range.')
    fig,ax=new(); vals=[.08025787482217676,.134311598,1,6.18780922259,2.64862017206]
    ax.barh(['Sphere weight','Idealized MRL normal load','Bench upper load','Old invalid normal-load sum','Old invalid residual force'],vals,color=['#2563eb','#2563eb','#64748b','#b91c1c','#b91c1c']); ax.set(xlabel='Force magnitude (N)'); ax.invert_yaxis()
    save(fig,4,'Task scale and rejected-regression scale are distinct','The old deep-overlap state is not evidence for physical contact geometry or receiver feasibility.')
    fig,ax=new(); ax.plot([r['load_n'] for r in legacy['curves']],force,'o-',label='Measured steady'); ax.plot([0,1],[0,1],'k--',label='Applied total load'); ax.legend(); ax.set(xlabel='Applied total normal load (N)',ylabel='Measured steady normal force (N)')
    save(fig,5,'Monotonic load balance does not establish acceptable deformation','No negative normal forces or positive-held-load contact losses in the frozen bench.')
    fig,ax=new()
    for r in source['rows']:
        if r['cycle']:
            tr=trace(r); ax.plot(b.column(tr,'signed_overlap_m')*1000,b.column(tr,'normal_force_n'),label=f"{r['load_n']:.3g} N")
    ax.set(xlabel='Signed overlap (mm; negative = separated)',ylabel='Contact normal force (N)',xlim=(-.1,1.4)); ax.legend()
    save(fig,6,'Load-unload cycles show dissipative hysteresis','Same 0.4-s ramps up/down. The zero-load free-drift tail extends beyond the displayed -0.1-mm range.')
    fig,axes=plt.subplots(1,2,figsize=(12,6.4)); row=next(r for r in source['rows'] if r['cycle'] and r['load_n']==.5); tr=trace(row); t=b.column(tr,'time_s')
    for key,label in [('external_work_j','External work'),('gravity_work_j','Gravity work'),('contact_work_j','Contact work'),('kinetic_energy_j','Kinetic energy')]: axes[0].plot(t,b.column(tr,key)*1e6,label=label)
    axes[0].set(xlabel='Time (s)',ylabel='Energy / work (microjoules)'); axes[0].legend(fontsize=8)
    axes[1].plot(t,b.column(tr,'work_balance_residual_j')*1e6); axes[1].set(xlabel='Time (s)',ylabel='Work-balance remainder (microjoules)')
    save(fig,7,'Energy accounting: finite-step remainder remains explicit','0.50-N cycle. No exact continuum contact-energy equivalence or global passivity proof is claimed.')
    fig,axes=plt.subplots(1,2,figsize=(12,6.4))
    for load in p.config()['bench']['sensitivity_loads_n']:
        cases=sorted([r for r in legacy['sensitivity'] if r['load_n']==load and not r['solver_diagnostic']],key=lambda r:r['timestep_s'])
        axes[0].plot([r['timestep_s']*1000 for r in cases],[r['penetration_m']*1000 for r in cases],'o-',label=f'{load:.3g} N')
        axes[1].plot([r['timestep_s']*1000 for r in cases],[r['settling_after_ramp_seconds'] for r in cases],'o-',label=f'{load:.3g} N')
    axes[0].set(xlabel='Timestep (ms)',ylabel='Steady overlap (mm)'); axes[1].set(xlabel='Timestep (ms)',ylabel='Settling after ramp (s)'); axes[0].legend(fontsize=8)
    save(fig,8,'Timestep changes transients, not the settled legacy contact law','Physical ramp and hold durations are fixed. Settling uses an engineering reporting band, not a success threshold.')
    fig,ax=new(); cases=[r for r in legacy['sensitivity'] if r['solver_diagnostic']]; ax.bar([str(round(r['load_n'],3)) for r in cases],[r['delta_difference_m']*1e9 for r in cases]); ax.set(xlabel='Load (N)',ylabel='Tighter solver minus baseline overlap (nm)')
    save(fig,9,'Tighter solver does not resolve the legacy compliance','400 iterations / 1e-12 versus 100 / 1e-8. This isolates numerical convergence, not grasp performance.')
    fig,ax=new(); ax.plot(force,[r['secant_stiffness_npm'] for r in legacy['curves']],'o-',label='Secant F/delta'); ax.plot(force,[r['gradient_stiffness_npm'] for r in legacy['curves']],'s-',label='Local finite-difference dF/ddelta'); ax.legend(); ax.set(xlabel='Normal force (N)',ylabel='Effective numerical stiffness (N/m)')
    save(fig,10,'Effective stiffness is a numerical descriptor, not Young modulus','No material constants were invented. Force-displacement calibration requires PI material data or an approved approximation scope.')
    fig,ax=new(); ax.axis('off'); labels=['Legacy','TC10','IMP99','TC10 + IMP99']; rows=[]
    for x in summary['suites']:
        c=x['candidate'] or {'solref':[.02,1],'solimp':[.9,.95,.001,.5,2]}
        rows.append([str(c['solref']),str(c['solimp']),'Unchanged','Not approved' if x['candidate'] else 'Legacy'])
    table=ax.table(cellText=rows,rowLabels=labels,colLabels=['solref','solimp','Friction','Status'],loc='center',colWidths=[.18,.42,.17,.18]); table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,2.6)
    save(fig,11,'Three interpretable contact options; no production selection','Reference time constant and impedance are varied independently and together. No task-based parameter tuning.')
    fig,ax=new()
    for label,x in zip(labels,summary['suites']): ax.plot([r['normal_force_n'] for r in x['curves']],[r['overlap_m']*1000 for r in x['curves']],'o-',label=label)
    ax.set(xlabel='Measured normal force (N)',ylabel='Steady overlap (mm)'); ax.legend()
    save(fig,12,'Independent force-deformation comparison across all options','Each option received the identical 36-run suite. Smaller deformation alone does not select physical truth.')
    fig,axes=plt.subplots(1,2,figsize=(12,6.4))
    for label,x in zip(labels,summary['suites']):
        cases=sorted([r for r in x['sensitivity'] if r['load_n']==1. and not r['solver_diagnostic']],key=lambda r:r['timestep_s'])
        axes[0].plot([r['timestep_s']*1000 for r in cases],[r['penetration_m']*1000 for r in cases],'o-',label=label)
        axes[1].plot([r['timestep_s']*1000 for r in cases],[r['settling_after_ramp_seconds'] for r in cases],'o-',label=label)
    axes[0].set(xlabel='Timestep (ms)',ylabel='Steady overlap at 1 N (mm)'); axes[1].set(xlabel='Timestep (ms)',ylabel='Settling after ramp (s)'); axes[0].legend(fontsize=8)
    save(fig,13,'Candidate timestep robustness: steady and transient views','All positive solref time constants exceed twice the largest timestep. Production timestep is unchanged.')
    fig,ax=new(); ax.axis('off'); ax.text(.5,.70,'NO CANDIDATE SELECTED',ha='center',fontsize=23,color='#b91c1c',weight='bold'); ax.text(.5,.47,'Numerical options characterized; material/scope approval awaits PI.',ha='center',fontsize=12); ax.text(.5,.26,'PHYSICS_V1_RIGID_CONTACT and hand force control are not validated.',ha='center',fontsize=12)
    save(fig,14,'Selected physics validation: approval gate remains closed','Do not confuse a stable isolated numerical bench with a calibrated physical hand-object interface.')
    fig,ax=new(); ax.axis('off'); blocks=[('Legacy: CP-C for full-range rigid-contact use','Finite, monotonic, repeatable; overlap reaches 19.6% of radius at 1 N.'),('Options: bench evidence, not production approval','Three candidates retain friction and reduce deformation; no E/nu is supplied.'),('Old receiver: rejected, not reinterpreted','6.60-mm settled overlap and 33-weight residual are invalid receiver evidence.'),('Stop before hand or receiver dynamics','PI contact-option review -> gated force primitive -> only then consider P1.')]
    for i,(head,body) in enumerate(blocks): ax.text(0,.92-i*.24,head,weight='bold',fontsize=12); ax.text(0,.83-i*.24,body,fontsize=11)
    save(fig,15,'Physical admissibility precedes task success','No receiver search, weld release, handoff, shape comparison, skin, object B or RL. P0 remains uncommitted.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--reports-only',action='store_true'); args=parser.parse_args()
    summary=summarize(); reports(summary)
    if not args.reports_only: figures(summary)
    print('Wrote P0 summary/report'+('' if args.reports_only else ' and 16 figures'))
