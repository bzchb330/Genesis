from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"seqgrasp-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _plot(path,x,ys,title,ylabel,release_time,labels=None,step=False):
    fig,ax=plt.subplots(figsize=(8,4.5)); (ax.step if step else ax.plot)(x,ys,where="post" if step else None) if step else ax.plot(x,ys)
    ax.axvline(release_time,color="black",linestyle="--",linewidth=1,label="support release"); ax.set(title=title,xlabel="simulation time [s]",ylabel=ylabel); ax.grid(True,alpha=.25)
    if labels is not None: ax.legend([*labels,"support release"],fontsize="small",ncol=2)
    else: ax.legend(fontsize="small")
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def plot_diagnostics(arrays,metadata,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); t=arrays["time"]; release=metadata["support_release_event"]["support_release_time"]
    _plot(out/"object_a_height.png",t,arrays["object_position"][:,2],"Object A vertical position","z [m]",release)
    _plot(out/"object_a_vertical_velocity.png",t,arrays["object_linear_velocity"][:,2],"Object A vertical velocity","v-z [m/s]",release)
    _plot(out/"object_a_displacement.png",t,arrays["object_translational_displacement_after_release"],"Object A translational displacement from release","distance [m]",release)
    _plot(out/"object_a_orientation_change.png",t,arrays["object_orientation_change_after_release"],"Object A orientation change from release","angle [rad]",release)
    _plot(out/"total_contact_force.png",t,arrays["total_configured_fingertip_normal_force"],"Total configured-fingertip normal force","force [N]",release)
    _plot(out/"normal_force_per_finger.png",t,arrays["finger_total_normal_force_raw"],"Normal force per finger","force [N]",release,metadata["finger_order"])
    _plot(out/"binary_finger_contact.png",t,arrays["tactile_contact_flags"],"Binary finger-contact state","contact flag [1]",release,metadata["finger_order"],step=True)
    _plot(out/"active_finger_count.png",t,arrays["active_finger_count"],"Number of active fingers","count",release,step=True)
    _plot(out/"number_of_contacts.png",t,arrays["active_contact_count"],"Number of active contacts","count",release)
    _plot(out/"actuator_commands.png",t,arrays["actuator_controls"],"Actuator commands","torque [N m]",release,metadata["actuator_order"])
    _plot(out/"joint_positions.png",t,arrays["joint_positions"],"Joint positions","angle [rad]",release,metadata["joint_order"])
    _plot(out/"episode_phase.png",t,arrays["episode_phase"],"Episode phase instrumentation","phase index",release,step=True)
