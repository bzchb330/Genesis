# Phase 3C-0.8 forearm-axis audit

- Parent body: `rh_forearm`.
- Direct child body: `rh_wrist`.
- Child anchor offset in parent coordinates: `[0.01, 0.0, 0.21301]` m.
- Normalized longitudinal axis in parent coordinates: `[0.046894504837346265, 0.0, 0.9988998475403128]`.
- Axis in world coordinates at nominal configuration: `[0.9988998475403126, 0.0, 0.04689450483734625]`.
- Evidence: normalized compiled-source vector from the rh_forearm origin to the direct rh_wrist child anchor; this follows the 0.21301-m wrist assembly extent rather than assuming a world axis.

The runtime wrapper injects `forearm_PS` into the parsed Phase-3 scene composition. The official vendored Shadow Hand XML is read only. The -90 to +90 deg interval is an engineering diagnostic range for a robot forearm/manipulator mount, not a human-anatomy claim.
