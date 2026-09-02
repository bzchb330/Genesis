# Phase 3C-P0 current contact-physics audit

Baseline: `07a61ca7731b47f5b7bb03263cd956ad5c588b3e`.
MuJoCo 3.11.0; compiled model, not guessed XML defaults.

| Field | Active value | Provenance |
|---|---|---|
| Timestep | 0.002 s | Project Phase 3 config, composed by phase3/model.py |
| Integrator / solver | Euler / Newton | MuJoCo defaults |
| Iterations / tolerance | 100 / 1e-8 | MuJoCo defaults |
| Cone / impratio | elliptic / 10 | Shadow Hand Menagerie XML |
| Sphere condim / priority | 6 / 1 | Project object composition |
| Hand collision condim / priority | 3 / 0 | MuJoCo defaults |
| Sphere friction | [0.5, 0.01, 0.003] | Project object config |
| Hand geom friction | [1, 0.005, 0.0001] | MuJoCo defaults |
| Combined friction | [0.5, 0.5, 0.01, 0.003, 0.003] | Sphere priority wins |
| Sphere solref | [0.02, 1] | MuJoCo defaults |
| Sphere solimp | [0.9, 0.95, 0.001, 0.5, 2] | MuJoCo defaults |
| Hand solref | [0.005, 1] | Menagerie plastic default class |
| Hand solimp | [0.5, 0.99, 0.0001, 0.5, 2] | Menagerie plastic default class |
| Combined solref / solimp | Sphere values above | Sphere priority wins |
| Geom and combined margin / gap | 0 / 0 m | MuJoCo defaults |
| Sphere radius / density | 0.0125 m / 1000 kg/m^3 | phase3c07.py |
| Sphere mass / weight | 0.00818123086872342 kg / 0.08025787482217676 N | Compiled mass; gravity 9.81 m/s^2 |
| Sphere inertia | Three equal principal values 5.113269292952139e-7 kg m^2 | Compiled geometry |

The sphere's priority overrides the hand's normal-contact parameters. Thus the
bench plane carries the representative hand geom settings, while the sphere
retains its higher priority. Runtime contact fields are checked against a
shallow static sphere-hand probe. No receiver dynamics are executed for this audit.

Collision representations include hand capsules, cylinders, boxes and convex
collision meshes; the object is an analytic sphere. Every named collision geom,
its enum, size and parameter arrays are saved in the machine audit. Visual mesh
appearance is not the collision shape and does not specify elasticity.

Additional solver fields: line-search iterations 50, line-search tolerance 0.01,
no-slip iterations 0, no-slip tolerance 1e-6, CCD iterations 35, CCD tolerance
1e-6. Enable/disable flags are zero. The temporary acquisition weld is not a
contact constitutive law and is absent from the independent primary bench.

## Material provenance

**MATERIAL ELASTIC CONSTANTS NOT SPECIFIED BY PI**

The baseline's 459 tracked text/code/config/XML files were searched for Young's
modulus, Poisson ratio, elastic modulus, restitution and material/compliance
specifications. No matching material-elastic specification was found. A density
of 1000 kg/m^3 does not identify the sphere as a particular material. The
Menagerie `plastic` class is not a supplied constitutive law. Numerical damping
ratios and passive joint damping are not measured material damping constants.

| Needed for quantitative material calibration | Present? |
|---|---|
| Sphere material identity, Young's modulus, Poisson ratio | No |
| Hand surface material identity, Young's modulus, Poisson ratio | No |
| Surface-layer thickness and rigid-substrate assumptions | No |
| Restitution/viscoelastic loss versus rate, if modeled | No |
| Measured force-displacement data and uncertainty | No |
| Applicable deformation, load and rate range | No |

Hertz is retained only as the symbolic reference
`F = (4/3) E* sqrt(R) delta^(3/2)`, with
`1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2`. No numerical E or nu is supplied.
Measured `dF/d(delta)` and secant stiffness are numerical descriptors, not
identified material constants.

Machine evidence: `outputs/phase3CP0/current_physics.json`,
`material_audit.json`, `frozen_protocol.json` and
`preserved_phase3C12B_hashes.json`. Production source and historical outputs are
not changed by this audit.

Reference: [MuJoCo solver parameters and contact mixing rules](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters).
