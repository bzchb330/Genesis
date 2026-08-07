# Hand Swap Checklist

The Allegro Hand is a development placeholder. Replacing it should require configuration and asset changes, not Python joint/finger edits.

Update `configs/hand_allegro.yaml` (or provide an equivalent hand YAML):

1. `model_path` to the new MJCF and preserve its license/provenance beside the asset.
2. `dof_count`, ordered `actuator_names`, and matching ordered one-DoF `joint_names`.
3. `palm_body`, ordered `fingertip_bodies`, and `finger_geom_mapping` using stable collision geom names.
4. Fixed `mount_pos` and `mount_quat` for the new model.
5. Ensure actuators can be converted to torque motors by `scene_builder.py`, or extend configuration if the model requires a different documented actuator representation.
6. Replace the actuator-keyed diagnostic fractions in `configs/diagnostic_grasp_a.yaml`; these are diagnostic-only and must not be transferred blindly.
7. Revisit the diagnostic fixture pose because hand geometry changes. Do not treat it as an evaluation criterion.
8. Run `check_install.py`, placement checks, scripted diagnostics, and the full tests.

Python resolves joint/actuator addresses by configured names and validates counts against the model. No Python module contains Allegro actuator, joint, fingertip, or mesh names. The target 20-DoF hand is intentionally not designed here.
