"""Vector P0.5R figures from saved states; zero physics integration."""
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from seqgrasp import phase3cp05r as p

S=p.read('summary.json'); names=p.config()['candidates']; labels=['IMP99','TC10_IMP99']; colors=['#007c91','#cf642a']
DEST=p.ROOT/'docs/figures/phase3CP05R'; made=[]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
primary=[next(t for t in S['trials'] if t['physics_name']==n and t['nominal_dt_s']==.002) for n in names]
traces=[p.old.load_trace(t['trace']) for t in primary]
A=S['equilibrium_audit'][0]; Q=A['snapshot']; before=p.old.load_trace(p.read('equilibrium_states/'+names[0]+'.json')['history'])[:501]
after=p.old.load_trace(Q['restore_confirmation_trace'])


def finish(fig,name,title,subtitle):
    fig.suptitle(title,x=.08,ha='left',fontsize=15,fontweight='bold',y=.985)
    fig.text(.08,.915,subtitle,fontsize=9,color='#4b5660',va='top')
    fig.subplots_adjust(left=.1,right=.96,bottom=.17,top=.79,wspace=.35,hspace=.5)
    fig.text(.08,.025,'P0.5R | IMP99 / TC10_IMP99 | Numerical contact diagnostics; no grasp-success claim',fontsize=8,color='#59636c')
    DEST.mkdir(parents=True,exist_ok=True); fig.savefig(DEST/(name+'.pdf'),metadata={'Title':title,'Subject':'; '.join(names)})
    plt.close(fig); made.append((DEST/(name+'.pdf')).relative_to(p.ROOT).as_posix())


def card(name,title,lines):
    fig,ax=plt.subplots(figsize=(10,6)); ax.axis('off'); y=1.02
    for line in lines:
        line=textwrap.fill(line,108); ax.text(0,y,line,transform=ax.transAxes,va='top',fontsize=11,color='#213746'); y-=.1+.05*(len(line.splitlines())-1)
    finish(fig,name,title,'Production reset repaired; required simultaneous M/R/L and 4-ms validation remain incomplete.')


def main():
    fig,ax=plt.subplots(figsize=(10,5)); old=np.array(A['nominal']['qpos'])[:25]; new=np.array(Q['qpos_eq'])[:25]
    ax.plot(old,'o--',label='Nominal (not equilibrium)',markersize=3); ax.plot(new,'o-',label='Natural equilibrium',markersize=3)
    ax.set(xlabel='Compiled hand joint index (0-24)',ylabel='Joint position (rad)'); ax.legend()
    finish(fig,'nominal_vs_dynamic_equilibrium_pose','Natural equilibrium changes the hand configuration','Same original actuator targets; no pose reset. Maximum joint displacement = 0.499896 rad.')
    for filename,key,title,unit,log in [('startup_qacc_before_after','max_acceleration','Initial acceleration reduced by ~74,632 times','Maximum |qacc| (rad/s^2)',True),('startup_qvel_before_after','max_speed','Zero initial velocity did not prevent startup motion','Maximum |qvel| (rad/s)',True)]:
        fig,ax=plt.subplots(figsize=(10,5))
        ax.plot([x['time_s']-before[0]['time_s'] for x in before],[max(1e-12,x[key]) for x in before],label='Nominal hand-only start')
        ax.plot([x['time_s']-after[0]['time_s'] for x in after],[max(1e-12,x[key]) for x in after],label='Cached equilibrium confirmation')
        ax.set(xlabel='Elapsed time (s)',ylabel=unit,xlim=(0,.5),yscale='log'); ax.legend()
        finish(fig,filename,title,'Original production dynamics in both traces. Plot floor 1e-12 only for log display; nominal qvel(t=0) = 0.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,x,label in zip(axes,[A['nominal'],Q['final_diagnostic']],['Nominal','Equilibrium']):
        ax.plot(x['net_generalized_force'][:25],label='Net generalized force'); ax.plot(x['mass_times_acceleration'][:25],'--',label='M qacc'); ax.set(title=label,xlabel='Hand DOF',ylabel='Generalized torque (Nm)'); ax.legend(fontsize=8)
    finish(fig,'equilibrium_generalized_force_balance','Dynamic equation balance is not static equilibrium','Nominal net torque is nonzero even though M qacc equals the force balance. Equilibrium net torque is ~2.2e-7 Nm.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    axes[0].bar(range(21),Q['final_diagnostic']['ctrl_error']); axes[1].bar(range(21),Q['final_diagnostic']['actuator_force'],color='#007c91')
    axes[0].set(xlabel='Actuator index',ylabel='ctrl - realized coordinate (rad)'); axes[1].set(xlabel='Actuator index',ylabel='Holding actuator force (Nm)')
    finish(fig,'equilibrium_ctrl_error_and_holding_torque','Nonzero tracking error supplies holding torque','Re-centering ctrl to realized coordinates gives max qacc = 81.817244 rad/s^2 (no integration counterfactual).')
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(4)
    ax.bar(x-.18,A['nominal']['tendon_length'],.36,label='Nominal'); ax.bar(x+.18,Q['tendon_state']['length'],.36,label='Equilibrium')
    ax.set(xticks=x,xticklabels=['Index J0','Middle J0','Ring J0','Little J0'],ylabel='Fixed-tendon summed angle (rad)'); ax.legend()
    finish(fig,'tendon_state_before_after_settle','Tendon sums conceal distal-joint redistribution','Native J0 controls J2+J1, not their individual split. Opposite-sign creep required 77.784 s of natural settling.')
    fig,ax=plt.subplots(figsize=(10,5)); ds=A['fk_displacements']
    ax.bar(range(len(ds)),[d['position_displacement_m']*1000 for d in ds],color='#007c91')
    short_labels=[d['name'].replace('phase3_middle_rh_mf','M\n').replace('phase3_ring_rh_rf','R\n').replace('phase3_little_rh_lf','L\n').replace('_collision_0','') for d in ds]
    ax.set(xticks=range(len(ds)),xticklabels=short_labels,ylabel='Geom-center displacement (mm)'); ax.tick_params(axis='x',labelsize=7)
    finish(fig,'settled_geometry_displacement','Settling invalidates nominal contact placement','Exact cached-state FK. Maximum M/R/L geom displacements: 22.135 / 23.529 / 25.039 mm.')
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(3)
    ax.bar(x-.18,[.0400105,.0400087,.0399961],.36,label='P0.5 nominal geometry')
    ax.bar(x+.18,[d['gap_m']*1000 for d in S['geometry']['closest']],.36,label='P0.5R settled geometry'); ax.axhline(.4,color='gray',ls='--',label='Minimum clearance = 0.4 mm')
    ax.set(xticks=x,xticklabels=['Middle','Ring','Little'],ylabel='Distal signed separation (mm)'); ax.legend(fontsize=8)
    finish(fig,'old_40um_vs_new_positive_gap','New minimum clearance is not three equal tip gaps','Sphere translation only; settled hand unchanged. Individual gaps: 0.832937 / 0.400000 / 0.865137 mm.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for n,label,color in zip(names,labels,colors):
        for dt,style in [(.002,'-'),(.004,'--')]:
            trial=next(t for t in S['trials'] if t['physics_name']==n and t['nominal_dt_s']==dt); rr=p.old.load_trace(trial['trace']); rr=[r for r in rr if r['time_s']<=.252+1e-9]
            for ax,key,scale in zip(axes,['max_speed','max_acceleration'],[1000,1]): ax.plot([r['time_s']*1000 for r in rr],[r[key]*scale for r in rr],style,color=color,label=f'{label}, {dt*1000:g} ms')
    axes[0].axhline(1,color='gray',ls=':'); axes[1].axhline(.5,color='gray',ls=':')
    axes[0].set(xlabel='Pre-hold elapsed (ms)',ylabel='Max speed (mrad/s)'); axes[1].set(xlabel='Pre-hold elapsed (ms)',ylabel='Max acceleration (rad/s^2)'); axes[1].legend(fontsize=7)
    finish(fig,'zero_command_preramp_validation','No spontaneous sphere contact; 4-ms speed guard fails','All commands unchanged during pre-hold. 2 ms passes 252 ms; 4 ms reaches 1.201 mrad/s at 8 ms and stops.')
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(projection='3d'); g=S['geometry']; center=np.array(g['sphere_center_palm_m'])*1000
    s=p.setup_trial(names[0],.002); origin,rot=p.palm_transform(s); u=np.linspace(0,2*np.pi,25); v=np.linspace(0,np.pi,13)
    ax.plot_wireframe(center[0]+12.5*np.outer(np.cos(u),np.sin(v)),center[1]+12.5*np.outer(np.sin(u),np.sin(v)),center[2]+12.5*np.outer(np.ones_like(u),np.cos(v)),color='#aeb8bd',alpha=.5,linewidth=.5)
    for surface in p.MRL:
        pts=np.array([rot.T@(s.data.xanchor[j]-origin)*1000 for j in s.joint_ids[surface]]); ax.plot(*pts.T,'o-',label=surface)
    ax.set(xlabel='Palm x (mm)',ylabel='Palm y (mm)',zlabel='Palm z (mm)'); ax.legend(); ax.view_init(22,-60)
    finish(fig,'repaired_MRL_test_geometry','Settled-state diagnostic geometry','Joint-chain schematic, not a collision-mesh rendering. Sphere fixed; no receiver or handoff.')
    for rr,label,filename in zip(traces,labels,['IMP99_MRL_force_timeline','TC10IMP99_MRL_force_timeline']):
        fig,ax=plt.subplots(figsize=(10,5))
        for surface in p.MRL: ax.plot([r['time_s'] for r in rr],[sum(c['normal_force_n'] for c in r['contacts'] if c['surface']==surface) for r in rr],label=surface)
        for t in [.252,1.252]: ax.axvline(t,color='gray',ls='--',alpha=.6)
        ax.set(xlabel='Elapsed time (s)',ylabel='Normal force (N)'); ax.legend()
        finish(fig,filename,label+': genuine loaded ring-little response','Production dt 2 ms; complete 0.252 s pre-hold + 1 s ramp + 2 s hold. Middle did not engage.')
    fig,ax=plt.subplots(figsize=(10,5))
    for rr,label,color in zip(traces,labels,colors): ax.plot([r['time_s'] for r in rr],[r['maximum_penetration_m']*1000 for r in rr],label=label,color=color)
    ax.set(xlabel='Elapsed time (s)',ylabel='Maximum penetration (mm)'); ax.legend()
    finish(fig,'IMP99_vs_TC10_penetration','Loaded overlap: TC10_IMP99 is smaller','2-ms sustained R/L exposure; not the required simultaneous M/R/L comparison.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,rr,label in zip(axes,traces,labels):
        for surface in p.MRL: ax.step([r['time_s'] for r in rr],[int(surface in r['topology']) for r in rr],where='post',label=surface)
        ax.set(title=label,xlabel='Elapsed (s)',ylabel='Active contact (1=yes)',ylim=(-.05,1.05)); ax.legend(fontsize=8)
    finish(fig,'IMP99_vs_TC10_contact_topology','Persistent R/L topology, missing middle','Topology must not be relabeled as three-finger validation. No force-stop censoring at 1/2 ms.')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,rr,label in zip(axes,traces,labels):
        for surface in ['ring','little']:
            pts=[(r['time_s'],c) for r in rr for c in r['contacts'] if c['surface']==surface and c['normal_force_n']>1e-9]
            n0=np.array(pts[0][1]['inward_normal_world']); angles=[np.degrees(np.arccos(np.clip(np.array(c['inward_normal_world'])@n0,-1,1))) for _,c in pts]
            ax.plot([t for t,_ in pts],angles,label=surface)
        ax.set(title=label,xlabel='Elapsed (s)',ylabel='Angle from first normal (deg)'); ax.legend(fontsize=8)
    finish(fig,'IMP99_vs_TC10_contact_normal_drift','Contact normals throughout loaded integration','No detached-contact extrapolation. Detailed final-500-ms migration statistics are stored separately.')
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(labels,[t['steady']['normal_force_variance_n2'] for t in primary],color=colors); ax.set(ylabel='Final-500-ms total-force variance (N^2)')
    finish(fig,'IMP99_vs_TC10_force_variance','Force variability in a real sustained-loaded window','Production dt 2 ms. This is a descriptive R/L window, not a three-finger success score.')
    fig,axes=plt.subplots(1,3,figsize=(10,5))
    for n,label,color in zip(names,labels,colors):
        ts=[x for x in S['trials'] if x['physics_name']==n and x['steady'] is not None]
        for ax,key,scale,ylabel in zip(axes,['penetration_mean_m','normal_force_mean_n','normal_force_variance_n2'],[1000,1,1],['Overlap (mm)','Force (N)','Force variance (N^2)']):
            ax.plot([t['nominal_dt_s']*1000 for t in ts],[t['steady'][key]*scale for t in ts],'o-',label=label,color=color); ax.set(xlabel='dt (ms)',ylabel=ylabel,xticks=[1,2,4],xlim=(.8,4.3)); ax.text(.94,.5,'4 ms\nGATED',transform=ax.transAxes,ha='right',color='#9d3641',fontsize=9)
    axes[0].legend(fontsize=7)
    finish(fig,'multitimestep_loaded_contact_comparison','Loaded comparison available at 1/2 ms only','No steady values are invented for the 4-ms pre-hold rejection; no extrapolation to that timestep.')
    fig,ax=plt.subplots(figsize=(10,5)); legacy=p.old.read(p.old.OUTPUT/'summary.json'); oldpeak=[next(x for x in legacy['hand'] if x['physics_name']==n and x['dt_s']==.002)['peak_total_normal_force_n'] for n in names]; x=np.arange(2)
    ax.bar(x-.18,oldpeak,.36,label='P0.5 startup stop'); ax.bar(x+.18,[t['observed_peak_total_force_n'] for t in primary],.36,label='P0.5R complete R/L run'); ax.set(xticks=x,xticklabels=labels,ylabel='Observed peak total normal force (N)'); ax.legend()
    finish(fig,'startup_artifact_vs_true_contact_response','Startup artifact versus repaired diagnostic exposure','Reset AND sphere placement/directions changed: this is not a one-factor causal ablation.')
    card('PHYSICS_V1_selection_rationale_repaired','No production V1: required evidence still incomplete',[
        'Reset: original targets, gravity, damping and gains; 77.784 s natural settling + 0.5 s independent confirmation.',
        'Common candidate cache: identical integration state. No artificial qvel zeroing or ctrl re-centering.',
        'At 1/2 ms: complete physical-duration contact exposure; persistent ring and little, but no middle.',
        'At 4 ms: pre-hold speed guard fails at 8 ms before any contact command.',
        'IMP99 has larger overlap; TC10_IMP99 has higher force. Missing exposure prevents selecting either.',
        'No V1, no historical dynamic regression, no bounded-force primitive, receiver, shape, skin, B or RL.'])
    card('phase3CP05R_causal_summary','What was repaired, and what was not',[
        'Zero servo error was not equilibrium: initial qacc = 81.616851 rad/s^2 with qvel = 0.',
        'Natural settling retains nonzero target error and holding torque; qacc falls to 0.001094 rad/s^2.',
        'Distal tendon split drift lasts tens of seconds; nominal contact geometry moves by up to 25 mm.',
        'New settled sphere placement is collision-free, with minimum 0.4-mm clearance.',
        'The frozen gentle ramp supplies valid sustained R/L data, not the required three-finger topology.',
        'Common-state 4-ms startup compatibility is not solved. Gates and commands were not relaxed.',
        'Next action requires PI review. P0.5R remains uncommitted; no merge to main.'])
    p.save('figures.json',dict(physics_names=names,figures=made,count=len(made),physics_steps=0)); print('Generated',len(made),'vector PDFs')


if __name__=='__main__': main()
