"""Reusable raw physical-admissibility diagnostics, independent of task success.

No built-in publication thresholds. A passed engineering gate is not physical
validation; thresholds must have an explicit label and supplied provenance.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class EngineeringGates:
    label: str
    max_penetration_m: float | None = None
    max_penetration_radius_ratio: float | None = None
    max_normal_load_weight_ratio: float | None = None
    max_residual_force_n: float | None = None
    max_residual_torque_nm: float | None = None
    max_kinetic_energy_j: float | None = None
    max_saturation_fraction: float | None = None

    def __post_init__(self):
        if not self.label.strip(): raise ValueError('Engineering gate needs an explicit label')
        for k,v in asdict(self).items():
            if k!='label' and v is not None and (not np.isfinite(v) or v<0):
                raise ValueError('Engineering bounds must be finite and nonnegative')


def diagnose(*,radius_m,weight_n,penetration_m,normal_forces_n,contact_gravity_force_n,
             torque_nm,actuator_saturation_fraction=(),external_support_force_n=(0,0,0),
             external_support_torque_nm=(0,0,0),kinetic_energy_j=0.,environment_support=False,gates=None):
    if radius_m<=0 or weight_n<=0: raise ValueError('Positive object radius and weight required')
    normal=float(np.sum(normal_forces_n)); pen=float(max(0.,penetration_m))
    result=dict(maximum_penetration_m=pen,penetration_radius_ratio=pen/radius_m,
                penetration_diameter_ratio=pen/(2*radius_m),sum_normal_force_n=normal,
                sum_normal_force_weight_ratio=normal/weight_n,
                net_contact_gravity_force_n=np.asarray(contact_gravity_force_n).tolist(),
                residual_force_norm_n=float(np.linalg.norm(contact_gravity_force_n)),
                net_torque_nm=np.asarray(torque_nm).tolist(),residual_torque_norm_nm=float(np.linalg.norm(torque_nm)),
                maximum_actuator_saturation_fraction=float(max(actuator_saturation_fraction,default=0.)),
                external_support_force_n=np.asarray(external_support_force_n).tolist(),
                external_support_torque_nm=np.asarray(external_support_torque_nm).tolist(),
                kinetic_energy_j=float(kinetic_energy_j),environment_support=bool(environment_support))
    scalar_values=[v for v in result.values() if isinstance(v,(int,float))]
    arrays=[contact_gravity_force_n,torque_nm,external_support_force_n,external_support_torque_nm,normal_forces_n,
            actuator_saturation_fraction,[radius_m,weight_n,penetration_m,kinetic_energy_j]]
    finite=bool(np.isfinite(scalar_values).all() and all(np.isfinite(a).all() for a in map(np.asarray,arrays)))
    failures=[] if finite else ['NONFINITE_DIAGNOSTIC']
    mapping=dict(max_penetration_m='maximum_penetration_m',max_penetration_radius_ratio='penetration_radius_ratio',
                 max_normal_load_weight_ratio='sum_normal_force_weight_ratio',max_residual_force_n='residual_force_norm_n',
                 max_residual_torque_nm='residual_torque_norm_nm',max_kinetic_energy_j='kinetic_energy_j',
                 max_saturation_fraction='maximum_actuator_saturation_fraction')
    if gates:
        for field,key in mapping.items():
            limit=getattr(gates,field)
            if limit is not None and result[key]>limit: failures.append(field.upper())
    result.update(engineering_gate=None if gates is None else asdict(gates),violations=failures,
                  engineering_gate_passed=None if gates is None else not failures,
                  scientifically_validated=False)
    return result
