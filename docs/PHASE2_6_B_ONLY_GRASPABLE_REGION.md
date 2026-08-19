# Phase 2.6 B-only graspable regions

The dense geometry stage evaluated 10,000 cylinder centers derived from pairwise fingertip workspace envelopes. It found 7,104 centers with multi-finger surface access, 1,644 with at least 120 degrees of predicted contact opposition, and 1,144 with positive Ferrari-Canny evidence. Fifty diverse geometry-only poses were passed to an 8,192-candidate joint pose/trajectory search.

Three nominal pose/trajectory combinations satisfied every frozen B-acquisition condition through the complete 500-step unsupported hold. Their local plus-or-minus 1 mm, plus-or-minus 0.10 rad perturbation tests produced 34/60 successes (56.7%): 45%, 80%, and 45% by profile. Successful final contact topologies were middle+thumb, index+thumb, and index+middle+thumb.

The exact three local boxes are recorded in `configs/phase2_6_b_only_graspable_regions.yaml`. They are intentionally small and separate: available data demonstrate local acquisition, not graspability throughout the large envelope between them. The prior Phase 2 box at x=[0.055,0.065], y=[0.115,0.125], z=[0.215,0.225] m does not overlap these validated local boxes and remains an immutable historical negative control.

These regions use only dense B-only geometry, frozen Ferrari-Canny diagnostics, B-only dynamic positives, and local B-only perturbations. No A-grasp resource association or formal sequential outcome was used to define them.
