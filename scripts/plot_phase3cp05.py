"""Vector figures from saved diagnostics; no simulation or model selection."""
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from seqgrasp import phase3cp05 as p, contact_physics as c

DEST=p.ROOT/'docs/figures/phase3CP05'
COLORS={c.IMP99:'#007c91',c.TC10:'#cf642a',c.LEGACY:'#727780'}
LABEL={c.IMP99:'IMP99',c.TC10:'TC10_IMP99',c.LEGACY:'LEGACY'}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
S=p.read(p.OUTPUT/'summary.json'); names=p.config()['candidates']; made=[]


def finish(fig,name,title,subtitle):
    fig.suptitle(title,x=.08,ha='left',fontsize=15,fontweight='bold',y=.985)
    fig.text(.08,.915,subtitle,fontsize=9,color='#4b5660',va='top')
    fig.subplots_adjust(left=.1,right=.96,bottom=.16,top=.79,wspace=.35,hspace=.5)
    fig.text(.08,.025,'Phase 3C-P0.5 | PHYSICS_CANDIDATE_IMP99 / PHYSICS_CANDIDATE_TC10_IMP99',fontsize=7,color='#59636c')
    DEST.mkdir(parents=True,exist_ok=True)
    fig.savefig(DEST/(name+'.pdf'),metadata={'Title':title,'Subject':'; '.join(names),'Author':'Genesis contact diagnostics'})
    plt.close(fig); made.append(name+'.pdf')


def card(name,title,lines):
    fig,ax=plt.subplots(figsize=(10,6)); ax.axis('off'); y=1.02
    for line in lines:
        wrapped=textwrap.fill(line,110)
        ax.text(0,y,wrapped,transform=ax.transAxes,va='top',fontsize=11,color='#213746')
        y-=.09+.05*(len(wrapped.splitlines())-1)
    finish(fig,name,title,'Numerical near-rigid assumption only. No material calibration or manipulation-success claim.')


def dynamic(name,key,scale,title,ylabel):
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,n in zip(axes,names):
        for x in S['impact']:
            if x['physics_name']!=n or x['dt_s']!=.002: continue
            rows=p.load_trace(x['trace']); ax.plot([r['time_s']*1000 for r in rows],[r[key]*scale for r in rows],label=f"{x['height_m']*1000:g} mm drop")
        ax.set(xlim=(0,250),xlabel='Time (ms)',ylabel=ylabel,title=LABEL[n]); ax.legend(fontsize=8)
    finish(fig,name,title,'Production dt = 2 ms; gravity-only impacts. Initial and final states included.')


def hand_plot(name,extract,title,ylabel):
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,n in zip(axes,names):
        h=next(x for x in S['hand'] if x['physics_name']==n and x['dt_s']==.002); rows=p.load_trace(h['trace'])
        series=extract(rows)
        for label,y in series.items(): ax.plot([r['time_s']*1000 for r in rows],y,'o-',label=label,
                                              drawstyle='steps-post' if name=='MRL_contact_topology_timeline' else 'default')
        ax.axvline(h['duration_s']*1000,color='#a32336',linestyle='--',label='Force stop')
        ax.set(xlabel='Recorded time (ms)',ylabel=ylabel,title=LABEL[n],xlim=(-.15,4.5)); ax.legend(fontsize=8)
    finish(fig,name,title,'CENSORED STARTUP: stopped at 4 ms; planned 3 s ramp/hold was not reached. No extrapolation.')


def main():
    card('physics_candidates_overview','Three explicit physics versions',[
        'LEGACY: solref [0.02, 1]; solimp [0.9, 0.95, 0.001, 0.5, 2]. Historical reference only.',
        'IMP99: solref [0.02, 1]; solimp [0.99, 0.99, 0.001, 0.5, 2].',
        'TC10_IMP99: solref [0.01, 1]; identical constant-0.99 solimp.',
        'All candidates: runtime friction [0.5, 0.5, 0.01, 0.003, 0.003]; effective dim 6.',
        'Production: dt 2 ms; Newton / 100 iterations / tolerance 1e-8; Euler; elliptic cone; impratio 10.',
        '31 explicit hand-object pairs. Native geometry, gains, tendons and unrelated contacts preserved.',
        'Decision: NO FREEZE. Actual-hand simultaneous M/R/L validation was not established.'])
    fig,ax=plt.subplots(figsize=(10,5))
    for n in names:
        rows=[r for r in S['regression']['rows'] if r['physics_name']==n]
        ax.plot([r['load_n'] for r in rows],[r['mean']['overlap_m']*1000 for r in rows],'o-',label=LABEL[n],color=COLORS[n])
    ax.set(xlabel='Total normal load (N)',ylabel='Steady overlap (mm)'); ax.legend()
    finish(fig,'IMP99_vs_TC10IMP99_quasistatic_contact','P0 regression: all eight values reproduced exactly','Same 0.4 s ramp + 4 s hold; final 200 samples. Absolute difference from P0 = 0 m.')
    fig,axes=plt.subplots(1,3,figsize=(10,5))
    for n in names:
        rows=[r for r in S['regression']['rows'] if r['physics_name']==n]
        for ax,key,scale,label in zip(axes,['a_m','a_over_R','delta_over_R'],[1000,100,100],['Overlap radius a (mm)','a / R (%)','Overlap / R (%)']):
            ax.plot([r['load_n'] for r in rows],[r['overlap_geometry'][key]*scale for r in rows],'o-',label=LABEL[n],color=COLORS[n]); ax.set(xlabel='Load (N)',ylabel=label)
    axes[0].legend(fontsize=8)
    finish(fig,'overlap_geometry_at_task_loads','Sphere-plane overlap geometry','Exact a = sqrt(max(0, 2 R delta - delta^2)); R = 12.5 mm. No universal cutoff assumed.')
    dynamic('dynamic_single_contact_penetration','penetration_m',1000,'Impact overlap is larger than static overlap','Overlap (mm)')
    dynamic('dynamic_single_contact_force','normal_force_n',1,'Harder normal response raises impact force peaks','Normal force (N)')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,n in zip(axes,names):
        xs=[x for x in S['impact'] if x['physics_name']==n]
        ax.scatter([x['height_m']*1000 for x in xs],[x['events']['breaks'] for x in xs],color=COLORS[n]); ax.set(xlabel='Drop height (mm)',ylabel='Contact breaks',ylim=(-.1,1.1),title=LABEL[n])
        ax.text(.04,.86,'12 trials: one make, zero breaks each\n1 / 2 / 4 ms\nFinal contact episode right-censored',transform=ax.transAxes,fontsize=9)
    finish(fig,'dynamic_contact_chatter','No repeated contact creation/destruction in the drop tests','Chatter is descriptive. Upward recovery velocity while contact persists is not detached rebound.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,n in zip(axes,names):
        x=next(x for x in S['impact'] if x['physics_name']==n and x['dt_s']==.002 and x['height_m']==.005); rows=p.load_trace(x['trace']); t=[r['time_s']*1000 for r in rows]
        for key,label in [('mechanical_energy_j','Mechanical energy - initial'),('contact_work_j','Contact work'),('energy_residual_j','Residual')]:
            ax.plot(t,[(r[key]-(x['initial_energy_j'] if key=='mechanical_energy_j' else 0))*1e6 for r in rows],label=label)
        ax.set(xlim=(0,250),xlabel='Time (ms)',ylabel='Energy (microjoules)',title=LABEL[n]); ax.legend(fontsize=8)
    finish(fig,'dynamic_contact_energy','Impact energy: dissipative, with finite discretization residual','5 mm drop, dt 2 ms. Work uses trapezoidal sampled force/velocity; residual is not zero.')
    g=p.read(p.OUTPUT/'geometry.json'); scene=p.setup_hand(c.IMP99,.002)
    from seqgrasp.phase3c0 import palm_transform
    origin,rot=palm_transform(scene); fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(projection='3d')
    center=np.array(g['center_palm_m'])*1000; u=np.linspace(0,2*np.pi,25); v=np.linspace(0,np.pi,13)
    ax.plot_wireframe(center[0]+12.5*np.outer(np.cos(u),np.sin(v)),center[1]+12.5*np.outer(np.sin(u),np.sin(v)),center[2]+12.5*np.outer(np.ones_like(u),np.cos(v)),color='#aeb8bd',alpha=.5,linewidth=.5)
    for x in p.MRL:
        points=np.array([rot.T@(scene.data.xanchor[j]-origin)*1000 for j in scene.joint_ids[x]])
        ax.plot(*points.T,'o-',label=x)
        pt=next(z for z in g['distal_contacts'] if z['surface']==x)['closest_points'][3:]; pt=rot.T@(np.array(pt)-origin)*1000
        ax.scatter(*pt,s=40)
    ax.set(xlabel='Palm x (mm)',ylabel='Palm y (mm)',zlabel='Palm z (mm)'); ax.legend(loc='upper left'); ax.view_init(22,-60)
    finish(fig,'shadow_hand_three_finger_test_geometry','Frozen near-contact M/R/L configuration','Joint-chain schematic and closest surface points, not a mesh rendering. All initial gaps positive; distal gaps ~40 um.')
    hand_plot('MRL_multicontact_penetration',lambda rr:{'Sphere-hand overlap':[r['maximum_penetration_m']*1e6 for r in rr]},'Hand startup penetration','Maximum overlap (um)')
    hand_plot('MRL_multicontact_force_timeline',lambda rr:{x:[sum(z['normal_force_n'] for z in r['contacts'] if z['surface']==x) for r in rr] for x in p.MRL},'Hand startup normal forces','Normal force (N)')
    hand_plot('MRL_contact_topology_timeline',lambda rr:{x:[int(x in r['topology']) for r in rr] for x in p.MRL},'No simultaneous M/R/L contact before force stop','Active (1 = yes)')
    card('MRL_contact_normal_drift','Contact-normal drift: insufficient time coverage',[
        'At production dt, contact first appears in the final saved state at 4 ms for ring and little.',
        'Middle contact never appears before the force stop. Each active pair has only one sampled contact state.',
        'Computed displacement/angle from the first observation is therefore zero by construction.',
        'This is NOT evidence of stable normals, stable topology or zero migration under sustained preload.',
        'No steady window, 3 s comparison or valid multi-contact stability inference is available.'])
    hand_plot('MRL_actuator_force_response',lambda rr:{x:[float(np.max(np.abs(np.array(r['actuator_force'])[scene.actuator_ids[x]]))) for r in rr] for x in p.MRL},'Native actuator response during startup','Max absolute actuator force (Nm)')
    hand_plot('MRL_fixture_wrench',lambda rr:{'Force norm (N)':[np.linalg.norm(r['weld_force_world_n']) for r in rr],'Torque norm (Nm)':[np.linalg.norm(r['weld_torque_world_nm']) for r in rr]},'Fixed-sphere reaction wrench','Wrench component norms (see legend)')
    for filename,key,scale,ylabel,title in [
        ('multitimestep_penetration_comparison','peak_penetration_m',1e6,'Peak overlap (um)','Timestep comparison: censored startup overlap'),
        ('multitimestep_force_variance','total_force_variance_tail_n2',1,'Observed-window variance (N^2)','Timestep comparison: startup force variance')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for n in names:
            xs=[x for x in S['hand'] if x['physics_name']==n]; ax.plot([x['dt_s']*1000 for x in xs],[x[key]*scale for x in xs],'o-',color=COLORS[n],label=LABEL[n])
        ax.set(xlabel='Timestep (ms)',ylabel=ylabel,xticks=[1,2,4]); ax.legend()
        finish(fig,filename,title,'1 ms: stopped at 3 ms; 2/4 ms: stopped at 4 ms. Unequal censored windows; not steady robustness.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for n in names:
        xs=[x for x in S['hand'] if x['physics_name']==n]; t=[x['dt_s']*1000 for x in xs]
        axes[0].plot(t,[x['peak_total_normal_force_n'] for x in xs],'o-',color=COLORS[n],label=LABEL[n]); axes[1].plot(t,[x['simultaneous_mrl_tail_fraction'] for x in xs],'o-',color=COLORS[n],label=LABEL[n])
    axes[0].set(xlabel='dt (ms)',ylabel='Peak total contact force (N)'); axes[0].legend(); axes[1].set(xlabel='dt (ms)',ylabel='Simultaneous MRL fraction',ylim=(-.05,1.05))
    finish(fig,'IMP99_vs_TC10IMP99_multicontact_summary','All six runs stopped before multi-contact validation','Identical startup geometry and command schedule. Guards are low-force engineering stops, not publication criteria.')
    card('PHYSICS_V1_selection_rationale','PHYSICS_V1_NEAR_RIGID: not created',[
        'Gate A: isolated P0 regression passed exactly for both candidates.',
        'Gate B: 24 impacts completed; no repeated make/break chatter; no energy above initial energy.',
        'Gate C-D: six force-guard stops at 3-4 ms. No simultaneous M/R/L ramp/hold validation.',
        'Impact tradeoff: TC10_IMP99 lowers overlap but raises force peaks and discretization residual.',
        'Neither candidate has enough actual-hand evidence for a production freeze.',
        'No optional sliding/spin test, no receiver run, no parameter retuning after results.'])
    fig,ax=plt.subplots(figsize=(10,5))
    legacy=p.read(p.ROOT/'outputs/phase3CP0/LEGACY_PHASE3C_CONTACT_PHYSICS/results.json')['rows']
    lr=[next(x for x in legacy if not x['cycle'] and x['load_n']==load and x['timestep_multiplier']==1 and not x['solver_diagnostic']) for load in p.config()['regression_loads_n']]
    ax.plot([r['load_n'] for r in lr],[r['mean']['overlap_m']*1000 for r in lr],'o-',color=COLORS[c.LEGACY],label='LEGACY (historical)')
    for n in names:
        rr=[r for r in S['regression']['rows'] if r['physics_name']==n]; ax.plot([r['load_n'] for r in rr],[r['mean']['overlap_m']*1000 for r in rr],'o-',color=COLORS[n],label=LABEL[n]+' (not selected)')
    ax.set(xlabel='Load (N)',ylabel='Steady overlap (mm)'); ax.legend()
    finish(fig,'legacy_vs_selected_physics','Legacy versus candidates: no selected physics exists','Requested selected-physics comparison is unavailable. Legacy remains CP-C for the full near-rigid load range.')
    card('phase3CP05_decision_summary','P0.5 stop result: physics remains unfrozen',[
        'Evidence preserved: 8 regression runs, 24 impact runs, 6 censored hand starts.',
        'The shared native-position-control initialization is not a demonstrated low-force M/R/L regime.',
        'Maximum command increment before termination < 0.000001 actuator-coordinate units.',
        'The stops cannot isolate intrinsic contact instability from startup/control and gravity effects.',
        'Next: PI review of diagnostic initialization/control, then separately authorized contact validation.',
        'Receiver reconstruction, shape, skin, RL and object B remain gated. P0.6 not implemented.',
        'P0.5 is intentionally uncommitted; main was not merged or modified.'])
    p.save('figures.json',dict(physics_names=names,figures=[(DEST/f).relative_to(p.ROOT).as_posix() for f in made],count=len(made),data_source='outputs/phase3CP05/summary.json',physics_steps=0))
    print('Generated',len(made),'vector PDFs')


if __name__=='__main__': main()
