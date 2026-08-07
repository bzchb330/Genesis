from __future__ import annotations
from pathlib import Path
import json
import os
import tempfile
import numpy as np
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"seqgrasp-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _describe(values):
    a=np.asarray(values,dtype=float); return {"minimum":float(np.min(a)),"maximum":float(np.max(a)),"mean":float(np.mean(a))}

def _longest_true_duration(mask,dt):
    longest=current=0
    for value in mask:
        current=current+1 if value else 0; longest=max(longest,current)
    return float(longest*dt)

def summarize_run(run,cfg):
    a=run.arrays; release=run.metadata["support_release_event"]; index=int(np.searchsorted(a["time"],release["support_release_time"],side="right")); post=slice(index,None); contact=a["active_finger_count"][post]>0; dt=cfg.scene.timestep
    loss=np.flatnonzero(~contact); time_to_loss=None if not len(loss) else float(loss[0]*dt)
    resting=run.metadata["table_resting_center_z_m"]; forces=a["finger_total_normal_force_raw"][post]
    return {"seed":run.metadata["seed"],"profile_name":run.metadata["profile_name"],"support_release_time_s":release["support_release_time"],"position_before_release_m":release["before"]["object_position"],"position_after_release_m":release["after"]["object_position"],"velocity_after_release_m_per_s":release["after"]["object_linear_velocity"],"immediate_vertical_displacement_m":release["immediate_vertical_displacement_m"],"final_vertical_displacement_m":release["final_vertical_displacement_m"],"post_release_height_m":_describe(a["object_position"][post,2]),"post_release_vertical_displacement_m":_describe(a["object_displacement_after_release"][post,2]),"post_release_translational_drift_m":_describe(a["object_translational_displacement_after_release"][post]),"post_release_orientation_change_rad":_describe(a["object_orientation_change_after_release"][post]),"peak_normal_force_per_finger_N":dict(zip(run.metadata["finger_order"],np.max(forces,axis=0).astype(float))),"mean_normal_force_per_finger_N":dict(zip(run.metadata["finger_order"],np.mean(forces,axis=0).astype(float))),"simultaneously_contacting_fingers":_describe(a["active_finger_count"][post]),"total_fingertip_contact_duration_s":float(np.sum(contact)*dt),"longest_continuous_fingertip_contact_s":_longest_true_duration(contact,dt),"duration_above_table_resting_center_height_s":float(np.sum(a["object_position"][post,2]>resting)*dt),"time_until_first_fingertip_contact_loss_s":time_to_loss,"contact_before_release":release["before"]["tactile_contact_flags"],"contact_after_release":release["after"]["tactile_contact_flags"],"force_before_release_N":release["before"]["finger_total_normal_force_raw"],"force_after_release_N":release["after"]["finger_total_normal_force_raw"],"terminated_early":run.metadata["terminated_early"],"termination_reason":run.metadata["termination_reason"]}

def aggregate_summaries(runs,summaries):
    fingers=runs[0].metadata["finger_order"]; release_indices=[int(np.searchsorted(r.arrays["time"],r.metadata["support_release_event"]["support_release_time"],side="right")) for r in runs]
    heights=np.concatenate([r.arrays["object_position"][i:,2] for r,i in zip(runs,release_indices)]); vertical=np.concatenate([r.arrays["object_displacement_after_release"][i:,2] for r,i in zip(runs,release_indices)]); drift=np.concatenate([r.arrays["object_translational_displacement_after_release"][i:] for r,i in zip(runs,release_indices)]); orientation=np.concatenate([r.arrays["object_orientation_change_after_release"][i:] for r,i in zip(runs,release_indices)]); active=np.concatenate([r.arrays["active_finger_count"][i:] for r,i in zip(runs,release_indices)])
    peak={f:_describe([s["peak_normal_force_per_finger_N"][f] for s in summaries]) for f in fingers}; mean={f:_describe([s["mean_normal_force_per_finger_N"][f] for s in summaries]) for f in fingers}; loss=[s["time_until_first_fingertip_contact_loss_s"] for s in summaries if s["time_until_first_fingertip_contact_loss_s"] is not None]
    return {"scientific_labels_assigned":False,"seeds":[s["seed"] for s in summaries],"run_count":len(runs),"post_release_height_m":_describe(heights),"post_release_vertical_displacement_m":_describe(vertical),"post_release_translational_drift_m":_describe(drift),"post_release_orientation_change_rad":_describe(orientation),"peak_normal_force_per_finger_N":peak,"mean_normal_force_per_finger_N":mean,"simultaneously_contacting_fingers":_describe(active),"longest_continuous_fingertip_contact_s":_describe([s["longest_continuous_fingertip_contact_s"] for s in summaries]),"duration_above_table_resting_center_height_s":_describe([s["duration_above_table_resting_center_height_s"] for s in summaries]),"vertical_velocity_immediately_after_release_m_per_s":_describe([s["velocity_after_release_m_per_s"][2] for s in summaries]),"time_until_first_fingertip_contact_loss_s":None if not loss else _describe(loss),"runs_without_observed_contact_loss":len(summaries)-len(loss),"early_termination_reasons":[s["termination_reason"] for s in summaries if s["terminated_early"]]}

def _aggregate_plot(path,runs,key,column,title,ylabel):
    n=min(len(r.arrays["time"]) for r in runs); t=runs[0].arrays["time"][:n]; values=np.stack([(r.arrays[key][:n] if column is None else r.arrays[key][:n,column]) for r in runs]); fig,ax=plt.subplots(figsize=(8,4.5))
    for trace in values: ax.plot(t,trace,color="0.65",alpha=.45,linewidth=.8)
    mean=values.mean(axis=0); ax.plot(t,mean,color="tab:blue",linewidth=2,label="mean"); ax.fill_between(t,values.min(axis=0),values.max(axis=0),color="tab:blue",alpha=.15,label="min-max")
    release=runs[0].metadata["support_release_event"]["support_release_time"]; ax.axvline(release,color="black",linestyle="--",linewidth=1,label="support release"); ax.set(title=title,xlabel="simulation time [s]",ylabel=ylabel); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def plot_aggregate(runs,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    specs=[("object_position",2,"aggregate_object_height.png","Object A vertical position","z [m]"),("object_linear_velocity",2,"aggregate_vertical_velocity.png","Object A vertical velocity","v-z [m/s]"),("object_translational_displacement_after_release",None,"aggregate_translational_displacement.png","Translational displacement from release","distance [m]"),("object_orientation_change_after_release",None,"aggregate_orientation_change.png","Orientation change from release","angle [rad]"),("total_configured_fingertip_normal_force",None,"aggregate_total_normal_force.png","Total configured-fingertip normal force","force [N]"),("active_finger_count",None,"aggregate_active_fingers.png","Number of active fingers","count")]
    for key,column,name,title,ylabel in specs: _aggregate_plot(out/name,runs,key,column,title,ylabel)
    for key,name,title,ylabel in (("finger_total_normal_force_raw","aggregate_force_per_finger.png","Normal force per finger","force [N]"),("tactile_contact_flags","aggregate_binary_contact_per_finger.png","Binary contact occupancy across seeds","mean flag [1]")):
        fingers=runs[0].metadata["finger_order"]; n=min(len(r.arrays["time"]) for r in runs); t=runs[0].arrays["time"][:n]; fig,axes=plt.subplots(len(fingers),1,figsize=(8,2.4*len(fingers)),sharex=True); release=runs[0].metadata["support_release_event"]["support_release_time"]
        for j,(finger,ax) in enumerate(zip(fingers,np.atleast_1d(axes))):
            values=np.stack([r.arrays[key][:n,j] for r in runs]); ax.plot(t,values.T,color="0.7",alpha=.3,linewidth=.7); ax.plot(t,values.mean(axis=0),color="tab:blue",linewidth=2); ax.fill_between(t,values.min(axis=0),values.max(axis=0),color="tab:blue",alpha=.15); ax.axvline(release,color="black",linestyle="--",linewidth=1); ax.set_ylabel(f"{finger}\n{ylabel}"); ax.grid(True,alpha=.25)
        axes[0].set_title(title); axes[-1].set_xlabel("simulation time [s]"); fig.tight_layout(); fig.savefig(out/name,dpi=160); plt.close(fig)

def write_summary(path,runs,summaries,aggregate):
    Path(path).write_text(json.dumps({"runs":summaries,"aggregate":aggregate},indent=2),encoding="utf-8")
