# Phase 3C-0.9 contact accessibility audit

State: 8 dimensions - sphere position (3), local sphere rotation (3), and smooth contact chart (2). The representative sample is stored step `525` of `C08_C07_STATE_00050_F0_STATIC_OPTIMUM`. Modes are M0 dual rolling/no-slip; M1 index guide with smooth thumb migration; M2 thumb guide with smooth index migration; and M3 one unloaded contact with the guide plus gravity as external drift.

- `M0_DUAL_ROLLING`: Delta1 rank `1` (translation `1`), target projection `0.963036124`, Delta2 rank `1` (translation `1`), second-order target residual `0.269372276`, classification **CT-C**.
- `M1_INDEX_GUIDE_THUMB_MIGRATION`: Delta1 rank `2` (translation `2`), target projection `0.965404945`, Delta2 rank `3` (translation `2`), second-order target residual `0.000115811268`, classification **CT-C**.
- `M2_THUMB_GUIDE_INDEX_MIGRATION`: Delta1 rank `2` (translation `2`), target projection `0.99753087`, Delta2 rank `3` (translation `2`), second-order target residual `0.000471178725`, classification **CT-C**.
- `M3_UNLOADED_SINGLE_GUIDE_GRAVITY`: Delta1 rank `2` (translation `2`), target projection `0.965404945`, Delta2 rank `2` (translation `2`), second-order target residual `0.260755233`, classification **CT-C**.

Finite-difference brackets used steps 1e-3, 5e-4, and 2.5e-4. M1/M2 bracket differences decreased toward the finest result, while their cyclic checks showed approximately O(epsilon^2) net motion. However, brackets increased full state-space rank without increasing translational rank enough to contain the target within the frozen numerical rank tolerance. All modes are CT-C. The current topology is locally insufficient under these explicitly smooth models; this is not a global LARC claim for nonsmooth MuJoCo contacts. Nonholonomic cycling is demonstrated kinematically but is not sufficient in the desired direction. RL is not implied.
