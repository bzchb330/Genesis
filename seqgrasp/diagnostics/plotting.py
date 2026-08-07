from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"seqgrasp-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _plot(path,x,ys,title,ylabel,labels=None):
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(x,ys); ax.set(title=title,xlabel="simulation time [s]",ylabel=ylabel); ax.grid(True,alpha=.3)
    if labels is not None: ax.legend(labels,fontsize="small",ncol=2)
    fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)

def plot_diagnostics(arrays,metadata,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); t=arrays["time"]
    _plot(out/"object_a_height.png",t,arrays["object_position"][:,2],"Object A vertical position","z [m]")
    _plot(out/"total_contact_force.png",t,arrays["finger_total_normal_force_raw"].sum(axis=1),"Total fingertip normal force","force [N]")
    _plot(out/"normal_force_per_finger.png",t,arrays["finger_total_normal_force_raw"],"Normal force per finger","force [N]",metadata["finger_order"])
    _plot(out/"number_of_contacts.png",t,arrays["contact_count"],"Number of contacts","count")
    _plot(out/"actuator_commands.png",t,arrays["actuator_controls"],"Actuator commands","torque [N m]",metadata["actuator_order"])
    _plot(out/"joint_positions.png",t,arrays["joint_positions"],"Joint positions","angle [rad]",metadata["joint_order"])
    _plot(out/"episode_phase.png",t,arrays["episode_phase"],"Episode phase instrumentation","phase index")
