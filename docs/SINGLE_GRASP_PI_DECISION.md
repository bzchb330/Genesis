# Single-Grasp PI Decision Worksheet

This worksheet maps possible scientific criteria to already logged signals. It intentionally selects no criterion or value.

## A. Grasp acquisition

| PI question | Available measured signal |
|---|---|
| Is contact alone sufficient? | `tactile_contact_flags`, `active_finger_count`, `finger_contact_count` |
| Must the object unload from the table? | `table_clearance`, object pose, configured table/object geometry |
| Is vertical displacement required? | `object_position[:, 2]`, `object_displacement_after_release[:, 2]` |
| Is a minimum number of contacting fingers required? | `active_finger_count` and ordered `active_fingers` |
| Must evidence persist for a duration? | simulation `time` plus contiguous samples of any selected signal |

The PI must decide which evidence is necessary, how signals combine, and whether persistence is required. `is_grasp_acquired` remains unresolved.

## B. Unsupported retention

| PI question | Available measured signal |
|---|---|
| Must the object remain fully clear of the table? | `table_clearance` |
| What hold duration is required? | time since `support_release_time` |
| What translational drift is allowable? | `object_translational_displacement_after_release` and XYZ displacement |
| What rotational drift is allowable? | `object_orientation_change_after_release` and quaternion history |
| Is continuous multi-finger contact required? | per-finger flags/counts and contiguous-contact durations |
| Is temporary contact loss allowed? | time histories of `active_finger_count` and first/contiguous loss events |

The PI must define unsupported retention and any allowed interruptions. `is_object_retained` remains unresolved.

## C. Loss/drop

| Possible PI criterion | Available measured signal |
|---|---|
| Return to table support | `table_clearance`, object height, table geometry |
| Leave the hand | object/palm poses and configured fingertip contacts |
| Lose required finger contacts | `active_fingers`, `tactile_contact_flags` |
| Exceed displacement | object displacement from support release |
| Exceed velocity | object linear/angular velocity |
| Leave workspace | existing mechanically configured workspace check |

The PI must choose what constitutes loss and whether criteria are instantaneous or persistent. `is_object_lost` remains unresolved except for the existing mechanical workspace-exit termination.

## Acquisition versus retention

Acquisition describes evidence established while support is present or during closing. Retention describes object behavior after the explicitly logged support-release event. Contact at release does not by itself establish unsupported retention. The software keeps these concepts separate through three independent, nullable criterion interfaces.
