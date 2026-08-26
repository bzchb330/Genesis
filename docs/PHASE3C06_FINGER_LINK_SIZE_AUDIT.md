# Phase 3C-0.6 finger-link size audit

## Source and method

The audit reads the official Shadow Hand source `assets/hands/shadow_right/right_hand.xml` without modifying it. For index, middle, ring, and little fingers, the proximal joint-to-next-joint length is the norm of the child middle-body `pos`; the intermediate length is the norm of the child distal-body `pos`.

| finger | segment | parent body | child body | MJCF vector (m) | length (mm) |
|---|---|---|---|---|---:|
| index | proximal | `rh_ffproximal` | `rh_ffmiddle` | (0.0, 0.0, 0.045) | 45.000 |
| index | intermediate | `rh_ffmiddle` | `rh_ffdistal` | (0.0, 0.0, 0.025) | 25.000 |
| middle | proximal | `rh_mfproximal` | `rh_mfmiddle` | (0.0, 0.0, 0.045) | 45.000 |
| middle | intermediate | `rh_mfmiddle` | `rh_mfdistal` | (0.0, 0.0, 0.025) | 25.000 |
| ring | proximal | `rh_rfproximal` | `rh_rfmiddle` | (0.0, 0.0, 0.045) | 45.000 |
| ring | intermediate | `rh_rfmiddle` | `rh_rfdistal` | (0.0, 0.0, 0.025) | 25.000 |
| little | proximal | `rh_lfproximal` | `rh_lfmiddle` | (0.0, 0.0, 0.045) | 45.000 |
| little | intermediate | `rh_lfmiddle` | `rh_lfdistal` | (0.0, 0.0, 0.025) | 25.000 |

There is no unique single link length: the audited set contains four 45 mm proximal links and four 25 mm intermediate links. Following the PI-approved rule, `L_ref` is the median of all eight corresponding non-thumb lengths: `(25 + 45) / 2 = 35 mm`.

- `D0 = L_ref = 0.035000 m` (35 mm)
- `R0 = D0 / 2 = 0.017500 m` (17.5 mm)
- inherited material density: `1000.0 kg/m^3`
- sphere volume: `0.000022449298 m^3`
- analytic and compiled sphere mass: `0.022449297504 kg`

The previous ellipsoid remains the default Phase 3A/3C object; the sphere is a separate Phase 3C-0.6 runtime configuration.
