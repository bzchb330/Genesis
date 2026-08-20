# Phase 2W rationale

Phase 2T-R demonstrated that equal free-digit count and exact free-digit identity are feasible in both endpoint groups, and that index+thumb can dynamically grasp object B in isolation. Its demonstrated native positive B region, however, intersected the endpoint hand at the existing fixed mount orientation in every checked placement.

Phase 2W therefore treats one common **static** wrist/root orientation as an experimental degree of freedom. No wrist trajectory is simulated: each candidate endpoint is initialized directly at the candidate rigid mount orientation, with its finger configuration and object-A pose relative to the palm preserved, and is then released under unchanged world-fixed gravity for dynamic endpoint validation.

This experiment tests static post-reorientation feasibility only. Dynamic wrist planning and control remain proposed future work. It does not define a scalar resource metric, change object mass, simulate transfer, add finger gaiting, add object C, or train reinforcement learning.
