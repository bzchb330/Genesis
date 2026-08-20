# J PI decision evidence after Phase 2W

| Candidate descriptor | Units/type | Physical interpretation | Observed Phase 2W variation | Apparent redundancy | Relationship to future acquisition | Sim-to-real observability |
|---|---|---|---|---|---|---|
| Free digit count | integer | Number of digits below the load-bearing threshold | Fixed at 2 | Does not encode identity | Insufficient alone in Phase 2T | Tactile/contact estimation |
| Free digit identity | categorical set | Which digits remain available | Fixed at index+thumb | Not reducible to count | Phase 2T/T-R distinguishes acquisition topology | Tactile/contact estimation |
| Free-digit reachable workspace | m^3 plus spatial set | Collision-filtered index/thumb reach | F index 1.8125e-05, thumb 1.7875e-05; P index 2.2375e-05, thumb 2.2e-05 | Scalar volume loses location | Necessary but not sufficient for B-only retention | Kinematics plus scene perception |
| Opposition-capable workspace | m^3 plus spatial set | B centers admitting index/thumb opposition | 1.1e-05 at highest-ranked failed candidate | Related to reachable workspace, but adds topology | Produced collision-free candidates; did not guarantee dynamic retention | Kinematics and object-pose perception |
| Free-palm volume | m^3 | Unoccupied voxels in fixed palm box | F 0.00335152; P 0.00335132 | Nearly redundant under equal topology | Did not distinguish the groups strongly here | Hand/object pose reconstruction |
| Object COM relative to palm | m, signed vector/distance | Location of retained A relative to palm | Signed surface means F 0.0252955; P -0.000724333 | Not captured by scalar free-palm volume | Describes endpoint topology; future acquisition relation not identified here | Object pose and hand pose |
| Palm contact | binary/fraction/force | Whether and how persistently A loads the palm | F 0.000; P 1.000 | Related to COM/palm distance | Defines endpoint class; no Phase 2W formal outcome relation | Tactile array |
| Wrist orientation | normalized quaternion or SO(3) element | Rigid hand mount orientation in world | 93 coarse and 114 refined orientations tested | Not reducible to palm contact or digit identity | Changed overlap and common geometric access | Robot encoders/kinematics |
| Gravity relative to palm | m/s^2 vector | Load direction in the hand frame | Highest-ranked failed candidate `[3.7541244715015316, 3.4683587617200162, -8.373358761720018]` | Determined by wrist orientation when world gravity is fixed | Affected endpoint survival; acquisition relationship not formally identified | IMU plus kinematics |
| Ferrari-Canny epsilon | dimensionless under current normalization | Geometric force-closure margin | Two-contact mapped approximation remained 0; A endpoint epsilon varied | Partly related to contact topology | Did not identify dynamic B success in Phase 2W | Contact geometry/force estimation |

No descriptor is normalized, weighted, or collapsed into an arbitrary scalar. Categorical digit topology remains categorical.

**TODO(PI): define whether future resource representation should be scalar, vector, structured/topological, or task-conditioned.**
