# Phase 2CM fingertip geometry audit

No geometry is modified by Phase 2CM.

| finger | collision geoms | type | size | representation | limitation |
|---|---:|---|---|---|---|
| index | 1 | capsule | `[0.012, 0.01, 0.0]` | single primitive | Rigid capsule contact is resolved at MuJoCo contact points; it is not a deformable distributed fingertip patch or taxel array. |
| middle | 1 | capsule | `[0.012, 0.01, 0.0]` | single primitive | Rigid capsule contact is resolved at MuJoCo contact points; it is not a deformable distributed fingertip patch or taxel array. |
| ring | 1 | capsule | `[0.012, 0.01, 0.0]` | single primitive | Rigid capsule contact is resolved at MuJoCo contact points; it is not a deformable distributed fingertip patch or taxel array. |
| thumb | 1 | capsule | `[0.012, 0.008, 0.0]` | single primitive | Rigid capsule contact is resolved at MuJoCo contact points; it is not a deformable distributed fingertip patch or taxel array. |
