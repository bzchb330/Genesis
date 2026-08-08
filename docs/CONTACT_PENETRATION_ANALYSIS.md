# Object-A Contact Penetration Analysis

This is an engineering-only MuJoCo contact characterization. Negative geom distance is reported as penetration depth in metres; it is not a grasp-validity threshold. All values below come from 20 fixed seeds per profile with unchanged scene physics.

## Existing candidates

| Source profile | Maximum penetration, mean (range) [m] | Steady-hold penetration, mean (range) [m] | Deepest finger | Relaxation after release |
|---|---:|---:|---|---:|
| `grasp_A_candidate_01` | 0.018883 (0.017623–0.020006) | 0.005849 (0.003934–0.006972) | index 6, middle 14 | 17/20 |
| `grasp_A_candidate_02` | 0.008364 (0.005202–0.012745) | 0.001438 (0.001339–0.002242) | middle 20 | 20/20 |
| `grasp_A_candidate_03` | 0.011639 (0.010776–0.012830) | 0.004166 (0.004041–0.004248) | middle 20 | 20/20 |
| `grasp_A_candidate_05` | 0.014901 (0.013165–0.017191) | 0.006573 (0.004921–0.007214) | index 19, thumb 1 | 20/20 |

Penetration is therefore not only a fixture-release transient: every source retains nonzero steady-hold penetration. In most runs the deepest penetration relaxes after support release, but candidate 01 does not do so in three seeds.

## Fixed-physics local refinement

The search perturbed only posture fractions and diagnostic timing around each source. It screened 24 variants per source and validated the three best engineering-only candidates over the same 20 seeds. The temporary objective is explicitly `engineering_only`; it combines the pre-existing finite-window retention diagnostic with a penetration penalty and is neither metric J nor a scientific success rule.

| Source | Representative validated local variant | Mean maximum penetration [m] | Worst maximum penetration [m] | Change from source |
|---|---|---:|---:|---|
| 01 | `local_013` | 0.018669 | 0.020006 | Very small mean reduction; no worst-case reduction |
| 02 | `local_003` | 0.007011 | 0.010395 | Lower descriptive penetration |
| 03 | `local_013` | 0.010472 | 0.011428 | Lower descriptive penetration |
| 05 | `local_019` | 0.013214 | 0.014903 | Lower descriptive penetration |

The final resource profiles use source-distinct retained refinements selected by the configured combined diagnostic: source 01 `local_015`, source 02 `local_022`, and source 03 `local_013`. Source 01's selected combined-objective variant does not materially reduce penetration. No refinement eliminates penetration, and this report intentionally supplies no acceptance cutoff.

Raw per-seed data are generated at `outputs/multi_grasp_resource_probing/contact_penetration_originals.json` and `local_refinement.json`; `outputs/` is intentionally ignored by Git.
