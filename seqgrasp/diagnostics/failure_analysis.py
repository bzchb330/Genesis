from __future__ import annotations
import numpy as np

def analyze_contact_sequence(run,cfg)->dict:
    a=run.arrays; event=run.metadata["support_release_event"]; release=int(np.searchsorted(a["time"],event["support_release_time"],side="right")); post=np.arange(release,len(a["time"])); fingers=run.metadata["finger_order"]
    first_loss={}
    for index,finger in enumerate(fingers):
        present=a["finger_object_contact_count"][post,index]>0; lost=np.flatnonzero(~present); first_loss[finger]=None if not len(lost) else float(lost[0]*cfg.scene.timestep)
    counts=a["finger_object_contact_count"][release]; normals=a["finger_object_contact_normal_world"][release]; active=np.flatnonzero(counts>0); dots={}
    for i in range(len(active)):
        for j in range(i+1,len(active)):
            left,right=int(active[i]),int(active[j]); dots[f"{fingers[left]}:{fingers[right]}"]=float(np.dot(normals[left],normals[right]))
    return {"contacting_fingers_immediately_after_release":[fingers[i] for i in active],"never_contacting_fingers":[fingers[i] for i in range(len(fingers)) if not np.any(a["finger_object_contact_count"][:,i]>0)],"first_contact_loss_after_release_s":first_loss,"contact_positions_immediately_after_release_m":dict(zip(fingers,a["finger_object_contact_position_world"][release].tolist())),"inward_contact_normals_immediately_after_release":dict(zip(fingers,normals.tolist())),"pairwise_contact_normal_dot":dots,"object_normal_force_immediately_after_release_N":dict(zip(fingers,a["finger_object_normal_force_raw"][release].astype(float))),"all_fingertip_force_immediately_after_release_N":dict(zip(fingers,a["finger_total_normal_force_raw"][release].astype(float))),"minimum_table_clearance_m":float(np.min(a["table_clearance"][release:])),"first_table_contact_after_release_s":None if not np.any(a["object_table_contact"][release:]) else float(np.flatnonzero(a["object_table_contact"][release:])[0]*cfg.scene.timestep),"maximum_saturated_actuators":int(np.max(a["actuator_saturation_count"])),"maximum_joint_limit_excess_rad":float(np.max(a["maximum_joint_limit_excess_rad"])),"joint_limit_violation_observed":bool(np.any(a["joint_limit_violation"])),"joint_targets_hold_after_release":True,"final_vertical_displacement_m":float(event["final_vertical_displacement_m"])}
