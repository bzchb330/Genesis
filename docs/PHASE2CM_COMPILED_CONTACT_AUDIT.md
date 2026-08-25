# Phase 2CM compiled contact audit

This audit compiles the exact half-scale scene and wrist-transformed model used by the deterministic Phase 2W/2H replay path. Values below are read from `mujoco.MjModel`, not inferred from source XML.

| role | geom_id | geom_name | body_id | body_name | geom_type | geom_size | condim | friction | solref | solimp | priority | solmix | contype | conaffinity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| index_fingertip | `12` | `ff_tip_collision` | `6` | `ff_tip` | capsule | `[0.012, 0.01, 0.0]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |
| middle_fingertip | `22` | `mf_tip_collision` | `11` | `mf_tip` | capsule | `[0.012, 0.01, 0.0]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |
| ring_fingertip | `32` | `rf_tip_collision` | `16` | `rf_tip` | capsule | `[0.012, 0.01, 0.0]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |
| thumb_fingertip | `42` | `th_tip_collision` | `21` | `th_tip` | capsule | `[0.012, 0.008, 0.0]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |
| palm | `1` | `(unnamed)` | `1` | `palm` | mesh | `[0.02096313234957383, 0.06375550839492264, 0.07001490886509427]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `0` | `0` |
| palm | `2` | `(unnamed)` | `1` | `palm` | box | `[0.0204, 0.0565, 0.0475]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |
| object_B | `44` | `object_b_geom` | `23` | `object_b` | cylinder | `[0.0125, 0.02, 0.0]` | `3` | `[0.8, 0.01, 0.001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |
| table | `0` | `table` | `0` | `world` | box | `[0.5, 0.5, 0.02]` | `3` | `[1.0, 0.005, 0.0001]` | `[0.02, 1.0]` | `[0.9, 0.95, 0.001, 0.5, 2.0]` | `0` | `1.0` | `1` | `1` |

## Runtime contact audit

A deterministic SHA-256-ordered sample of 3 eligible Phase 2H trials produced 10796 index–B/thumb–B contact records. Every record, including contact frame and the six-value `mj_contactForce` buffer, is stored at `outputs/phase2CM/contact_model_audit/runtime_contacts.jsonl`.

- actual runtime index–B contact dimensions: `[3]`
- actual runtime thumb–B contact dimensions: `[3]`
- sample trial IDs: `phase2W-static-wrist-B-only:75192960149548b7d10b6fd5f541f1a7d5b862ddadb98e773d6aee9906e13b79`, `phase2W-static-wrist-B-only:089e72238b466dec279880ff4617a00572548ccdbf610caa45cbbd430ccc1402`, `phase2W-static-wrist-B-only:30055edcd8911369b589f360fe236e2301b7f3b2bf08a378fa9db351fdaddbf0`
