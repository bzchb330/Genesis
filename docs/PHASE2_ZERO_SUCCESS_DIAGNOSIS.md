# Phase 2 original zero-success diagnosis

The original formal experiment remains immutable under experiment ID `phase2_original_zero_success`. Its ignored raw records remain in `outputs/phase2/grasp_dataset/fcc7835446a4/correlation/formal/` and are never reopened for writing by Phase 2.5.

The batch contained 4,540 trials: `BOTH_RETAINED=0`, `A_DROPPED=0`, `B_NOT_ACQUIRED=2700`, `BOTH_LOST=1834`, and `INVALID=6`. Among 4,534 valid trials, the BOTH_RETAINED rate was 0%. The geometry preflight was nondegenerate: 1,439/4,000 representative grasp-pose pairs (35.975%) were geometrically reachable, 195/200 B poses were reachable for at least one representative grasp, and five remained challenging.

Every valid dynamic trial ended without final B-hand support, without final free-finger support, and with B contacting the table. The scripted motion therefore failed to convert geometric reachability into functional unsupported acquisition. With no positive dependent-variable class, raw, standardized, and grasp-clustered logistic regression—and McFadden pseudo-R²—were mathematically unidentifiable.

The Phase 2 dataset is retained as an engineering negative result showing that geometric reachability alone did not guarantee functional second-object acquisition. It is not evidence against the resource-correlation hypothesis, because that hypothesis was not testable without outcome variation.

Phase 2.5 uses separate experiment IDs, seeds, controllers, and output stores. It does not overwrite, reclassify, mix, or continue the original formal records. Scalar J remains undefined.
