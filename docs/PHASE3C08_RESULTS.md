# Phase 3C-0.8 results

## Outcome

Classification: **KR-A**. The runtime forearm DOF reduced the native median residual from `49.158791` deg to `8.53773646e-07` deg; all 50 states were below 10 deg. Zero-angle equivalence passed over 50 states with a maximum recorded error of `0` at a `1e-12` tolerance.

The outcome-independent targeted manifest froze 10 states and five configurations per state before dynamics (`389e221947974a6de74b876a900cadbf20e27f66f448a6f29d15b7c01359c0e2`). F0 static orientation produced `0/20` entries; F1 coordinated orientation produced `0/30`; total `0/50`. The closest approach was `0.00422563655` m. Ring/little/palm-root contact counts were `0/1/0`; sphere loss was `7`; corridor-clear trials were `49`.

Maximum raw penetration by surface was `{'thumb': 0.001590207228863794, 'index': 0.0025292776079898875, 'middle': 0.0, 'ring': 0.0, 'little': 0.0025524958874267763, 'palm': 0.0}` m. No acceptability decision is inferred from these raw values.

## Causal interpretation

Forearm rotation causally changed the upstream gravity-orientation reachability from a roughly 49-deg median mismatch to effectively zero, but did not change dynamic pocket entry from the historical `0/500` to a nonzero outcome. Gravity orientation was therefore necessary but not sufficient. Additional whole-hand orientation is not indicated by this kinematic result; global hand translation remains a candidate diagnostic, not an implemented change.

The exact recommended next phase is a thumb/index in-hand transport Jacobian and lateral object-controllability audit, including rolling/sliding mechanics, with any global-translation comparison left for PI selection. Cage formation, skin/compliance changes, object B, and RL remain premature.

## Truthful render-only videos

- `outputs/phase3C08/videos/forearm_reorientation_retaining_sphere.mp4`
- `outputs/phase3C08/videos/best_targeted_pocket_transport.mp4`
- `outputs/phase3C08/videos/representative_failure.mp4`
