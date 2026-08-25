# Phase 3A Shadow Hand Audit

## Provenance

- Model: right Shadow Hand E3M5
- Official source: `https://github.com/google-deepmind/mujoco_menagerie.git`
- Vendored commit: `c1a4eeb85694ae1dffe33ff1797d4e528928a133`
- Upstream license: Apache-2.0 (vendored alongside the model)
- The upstream MJCF and assets are unchanged. Semantic collision names are added to the in-memory runtime XML.

## Compiled structure

- bodies: 28
- hand_bodies_excluding_world: 25
- hand_joints: 24
- scene_joints_including_object_freejoint: 25
- scene_dofs_including_object_freejoint: 30
- actuators: 20
- tendons: 4
- geoms: 64
- Palm body: `rh_palm`
- Wrist joints: `rh_WRJ2`, `rh_WRJ1`

## Semantic finger chains

- thumb: bodies `rh_thbase`, `rh_thproximal`, `rh_thhub`, `rh_thmiddle`, `rh_thdistal`; joints `rh_THJ5`, `rh_THJ4`, `rh_THJ3`, `rh_THJ2`, `rh_THJ1`; tip `rh_thdistal`
- index: bodies `rh_ffknuckle`, `rh_ffproximal`, `rh_ffmiddle`, `rh_ffdistal`; joints `rh_FFJ4`, `rh_FFJ3`, `rh_FFJ2`, `rh_FFJ1`; tip `rh_ffdistal`
- middle: bodies `rh_mfknuckle`, `rh_mfproximal`, `rh_mfmiddle`, `rh_mfdistal`; joints `rh_MFJ4`, `rh_MFJ3`, `rh_MFJ2`, `rh_MFJ1`; tip `rh_mfdistal`
- ring: bodies `rh_rfknuckle`, `rh_rfproximal`, `rh_rfmiddle`, `rh_rfdistal`; joints `rh_RFJ4`, `rh_RFJ3`, `rh_RFJ2`, `rh_RFJ1`; tip `rh_rfdistal`
- little: bodies `rh_lfmetacarpal`, `rh_lfknuckle`, `rh_lfproximal`, `rh_lfmiddle`, `rh_lfdistal`; joints `rh_LFJ5`, `rh_LFJ4`, `rh_LFJ3`, `rh_LFJ2`, `rh_LFJ1`; tip `rh_lfdistal`

## Joint limits and passive parameters

| Joint | Range (rad) | Damping | Armature | Friction loss |
|---|---:|---:|---:|---:|
| `rh_WRJ2` | [-0.523599, 0.174533] | 0.5 | 0.0002 | 0.01 |
| `rh_WRJ1` | [-0.698132, 0.488692] | 0.5 | 0.0002 | 0.01 |
| `rh_FFJ4` | [-0.349066, 0.349066] | 0.05 | 0.0002 | 0.01 |
| `rh_FFJ3` | [-0.261799, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_FFJ2` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_FFJ1` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_MFJ4` | [-0.349066, 0.349066] | 0.05 | 0.0002 | 0.01 |
| `rh_MFJ3` | [-0.261799, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_MFJ2` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_MFJ1` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_RFJ4` | [-0.349066, 0.349066] | 0.05 | 0.0002 | 0.01 |
| `rh_RFJ3` | [-0.261799, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_RFJ2` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_RFJ1` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_LFJ5` | [0.0, 0.785398] | 0.05 | 0.0002 | 0.01 |
| `rh_LFJ4` | [-0.349066, 0.349066] | 0.05 | 0.0002 | 0.01 |
| `rh_LFJ3` | [-0.261799, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_LFJ2` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_LFJ1` | [0.0, 1.5708] | 0.05 | 0.0002 | 0.01 |
| `rh_THJ5` | [-1.0472, 1.0472] | 0.05 | 0.0002 | 0.01 |
| `rh_THJ4` | [0.0, 1.22173] | 0.05 | 0.0002 | 0.01 |
| `rh_THJ3` | [-0.20944, 0.20944] | 0.05 | 0.0002 | 0.01 |
| `rh_THJ2` | [-0.698132, 0.698132] | 0.05 | 0.0002 | 0.01 |
| `rh_THJ1` | [-0.261799, 1.5708] | 0.05 | 0.0002 | 0.01 |

## Actuator limits

| Actuator | Control range | Force range | Position gain |
|---|---:|---:|---:|
| `rh_A_WRJ2` | [-0.523599, 0.174533] | [-10.0, 10.0] | 10 |
| `rh_A_WRJ1` | [-0.698132, 0.488692] | [-5.0, 5.0] | 8 |
| `rh_A_THJ5` | [-1.0472, 1.0472] | [-3.0, 3.0] | 0.4 |
| `rh_A_THJ4` | [0.0, 1.22173] | [-2.0, 2.0] | 1 |
| `rh_A_THJ3` | [-0.20944, 0.20944] | [-1.0, 1.0] | 0.5 |
| `rh_A_THJ2` | [-0.698132, 0.698132] | [-1.0, 1.0] | 1.5 |
| `rh_A_THJ1` | [-0.261799, 1.5708] | [-1.0, 1.0] | 1 |
| `rh_A_FFJ4` | [-0.349066, 0.349066] | [-1.0, 1.0] | 1 |
| `rh_A_FFJ3` | [-0.261799, 1.5708] | [-1.0, 1.0] | 1 |
| `rh_A_FFJ0` | [0.0, 3.1415] | [-1.0, 1.0] | 0.5 |
| `rh_A_MFJ4` | [-0.349066, 0.349066] | [-1.0, 1.0] | 1 |
| `rh_A_MFJ3` | [-0.261799, 1.5708] | [-1.0, 1.0] | 1 |
| `rh_A_MFJ0` | [0.0, 3.1415] | [-1.0, 1.0] | 0.5 |
| `rh_A_RFJ4` | [-0.349066, 0.349066] | [-1.0, 1.0] | 1 |
| `rh_A_RFJ3` | [-0.261799, 1.5708] | [-1.0, 1.0] | 1 |
| `rh_A_RFJ0` | [0.0, 3.1415] | [-1.0, 1.0] | 0.5 |
| `rh_A_LFJ5` | [0.0, 0.785398] | [-1.0, 1.0] | 1 |
| `rh_A_LFJ4` | [-0.349066, 0.349066] | [-1.0, 1.0] | 1 |
| `rh_A_LFJ3` | [-0.261799, 1.5708] | [-1.0, 1.0] | 1 |
| `rh_A_LFJ0` | [0.0, 3.1415] | [-1.0, 1.0] | 0.5 |

## Collision/contact representation

### thumb

- `phase3_thumb_rh_thdistal_collision_0`: compiled geom type 7, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]

### index

- `phase3_index_rh_ffdistal_collision_0`: compiled geom type 7, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]

### middle

- `phase3_middle_rh_mfdistal_collision_0`: compiled geom type 7, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]

### ring

- `phase3_ring_rh_rfdistal_collision_0`: compiled geom type 7, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]

### little

- `phase3_little_rh_lfdistal_collision_0`: compiled geom type 7, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]

### palm

- `phase3_palm_rh_palm_collision_0`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_1`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_2`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_3`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_4`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_5`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_6`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]
- `phase3_palm_rh_palm_collision_7`: compiled geom type 6, condim 3, friction [1.0, 0.005, 0.0001], solref [0.005, 1.0], solimp [0.5, 0.99, 0.0001, 0.5, 2.0]

## Solver

- timestep: 0.002 s
- cone enum: 1 (elliptic in upstream MJCF)
- impratio: 10.0
- integrator enum: 0
- iterations: 100
- line-search iterations: 50

No Phase 2 Allegro code path or historical physics parameter is modified by this integration.
