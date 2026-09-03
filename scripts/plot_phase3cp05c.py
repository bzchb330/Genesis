"""Vector figures from the static audit; never integrates dynamics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from seqgrasp import phase3cp05c as p

S=p.read('outputs/phase3CP05C/summary.json'); O=p.ROOT/'docs/figures/phase3CP05C'; O.mkdir(parents=True,exist_ok=True); made=[]
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})


def finish(fig,name,title,subtitle='Static audit; no contact, receiver, preload dynamics, or grasp-success criterion.'):
    fig.suptitle(title,x=.07,y=.98,ha='left',fontsize=15,fontweight='bold')
    fig.text(.07,.91,subtitle,color='#4b5660',fontsize=9)
    fig.subplots_adjust(left=.11,right=.97,bottom=.2,top=.78,wspace=.36)
    fig.text(.07,.025,'Phase 3C-P0.5C | Compiled Shadow Hand | CASE C: mandatory stop before preload',fontsize=8,color='#59636c')
    fig.savefig(O/(name+'.pdf'),metadata={'Title':title}); plt.close(fig); made.append(name+'.pdf')


def bar(name,title,labels,series,ylabel):
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(labels)); width=.75/len(series)
    for k,(legend,values) in enumerate(series.items()): ax.bar(x+(k-(len(series)-1)/2)*width,values,width,label=legend)
    ax.set(xticks=x,xticklabels=labels,ylabel=ylabel); ax.tick_params(axis='x',labelrotation=55,labelsize=7); ax.legend(fontsize=8)
    finish(fig,name,title)


def card(name,title,lines):
    fig,ax=plt.subplots(figsize=(10,5)); ax.axis('off')
    for i,line in enumerate(lines): ax.text(.02,.95-i*.13,line,transform=ax.transAxes,va='top',fontsize=11,color='#203747',wrap=True)
    finish(fig,name,title)


def main():
    joints=S['joints']; names=[x['joint_name'] for x in joints]
    bar('joint_drift_nominal_vs_settled','Natural settling redistributes the nominal pose',names,{'nominal':[x['q_nominal'] for x in joints],'settled':[x['q_settled'] for x in joints]},'Joint coordinate (rad)')
    cats=['direct','tendon coupled','unactuated']; vals=[sum(x['absolute_delta_q'] for x in joints if x['directly_actuated']),sum(x['absolute_delta_q'] for x in joints if x['tendon_coupled']),sum(x['absolute_delta_q'] for x in joints if x['unactuated'])]
    bar('joint_drift_by_actuation_type','Drift concentrates in coupled distal coordinates',cats,{'summed |drift|':vals},'Sum of absolute drift (rad)')
    A=np.asarray(S['moment_matrix']); fig,ax=plt.subplots(figsize=(10,6)); im=ax.imshow(A,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1); fig.colorbar(im,ax=ax,label='Moment coefficient'); ax.set(xlabel='Generalized-coordinate index',ylabel='Actuator index'); finish(fig,'actuator_transmission_matrix','Compiled actuator moment matrix is 21 x 25')
    fig,ax=plt.subplots(figsize=(10,5)); sv=S['allocation']['singular_values']; ax.semilogy(np.arange(1,len(sv)+1),sv,'o-'); ax.set(xlabel='Singular-value index',ylabel='Singular value',xticks=range(1,22,2)); finish(fig,'actuator_transmission_singular_values','Twenty-one nonzero singular values; four missing DOF directions')
    bar('required_static_generalized_force','MuJoCo inverse-dynamics holding force',names,{'required':[x['required_tau'] for x in joints]},'Generalized force (Nm)')
    an=[x['name'].replace('rh_A_','').replace('phase3c08_A_','') for x in S['actuators']]
    bar('unbounded_force_allocation','Least-norm unbounded actuator-force allocation',an,{'unbounded':S['allocation']['unbounded']['forces']},'Actuator force coordinate')
    bar('bounded_force_allocation','Existing limits do not change the least-squares solution',an,{'bounded':S['allocation']['bounded']['forces']},'Actuator force coordinate')
    bar('generalized_force_residual_by_dof','Residual is antisymmetric within distal tendon pairs',names,{'unbounded residual':[x['unbounded_residual'] for x in joints]},'Residual generalized force (Nm)')
    bar('actuator_force_utilization','No actuator force limit is approached',an,{'utilization':S['allocation']['bounded']['utilization']},'Absolute force-limit utilization')
    bar('required_vs_actuator_span_by_dof','Required generalized force versus actuator-span projection',names,{'required':[x['required_tau'] for x in joints],'represented':S['allocation']['unbounded']['represented_tau']},'Generalized force (Nm)')
    unavailable=[
      ('equilibrium_ctrl_preload','Equilibrium control preload was not constructed'),
      ('requested_vs_realized_actuator_force','No requested-versus-realized preload comparison'),
      ('startup_qacc_old_vs_preloaded','No preloaded startup simulation'),
      ('nominal_pose_drift_old_vs_preloaded','No preloaded nominal-pose drift simulation'),
      ('MRL_geometry_drift_old_vs_preloaded','No preloaded M/R/L geometry simulation'),
      ('equilibrium_force_balance','No preloaded dynamic equilibrium trace')]
    for name,title in unavailable: card(name,title,['CASE C: unbounded actuator allocation has a non-roundoff residual.','The protocol requires STOP; unavailable values are not fabricated.','No ctrl, kp, force limits, tendon model, damping, gravcomp, or qfrc_applied was changed.'])
    card('static_realizability_case_summary','CASE C - transmission limited',[f"A shape/rank: {S['allocation']['shape']} / {S['allocation']['rank']}",f"Residual: {S['allocation']['unbounded']['norm']:.12g} Nm ({100*S['allocation']['unbounded']['relative_residual']:.6f}% of required-force norm)",'Force limits inactive: bounded and unbounded solutions coincide.','Dominant residuals occur as opposite signs within distal fixed-tendon pairs.'])
    card('prior_geometry_implication_summary','Geometric results require a realizability qualifier',['Geometric calculations remain true for their sampled configurations.','This audit does not show that the nominal cup pose can be held exactly by the existing transmission.','Workspace fractions are preserved, not overwritten or rescanned.','Future realizability-aware confirmation is required before dynamic interpretation.'])
    card('foundational_exit_decision','Foundational exit gate did not pass',['CASE A was required; observed classification is CASE C.','No equilibrium preload, direct test, local perturbation, or physics V1 freeze occurred.','Required next action: PI interpretation/model decision.','B03, fly-by, bounded-force receiver, B and RL remain gated.'])
    p.save('figures.json',dict(count=len(made),figures=['docs/figures/phase3CP05C/'+x for x in made],physics_steps=0,local_perturbation_figure_omitted=True))
    print('Generated',len(made),'vector PDFs')


if __name__=='__main__': main()
