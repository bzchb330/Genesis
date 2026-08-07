# Determinism and Reproducibility

- Gymnasium: `env.reset(seed=seed)` initializes `self.np_random`.
- Object placement: receives that generator directly; the diagnostic uses `numpy.random.default_rng(seed)`.
- Action smoke test: seeds `action_space` independently.
- Scripted diagnostic: target schedule, fixture jitter, and initial placement all derive from its explicit seed.
- Python/NumPy utility: `seed_everything` exists for external runners that need global Python and NumPy state.

Tests compare complete trajectories for identical seeds and actions exactly on the same platform/runtime. Cross-platform physics comparisons should use a documented numerical tolerance; this is software tolerance, not a scientific evaluation threshold.
