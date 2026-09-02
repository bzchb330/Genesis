# Phase 3C-1.1 results

## Outcome

Phase 3C-1.1 reaches **CASE E** under the frozen protocol. Pair-isolated calibration confirmed measurable rigid-contact response, but 0/12 exact B03 states admitted a valid two-surface initializer. Cube and short-cylinder searches each froze six candidates, but none passed preload initialization. Consequently no Phase 3C-1.1 hold dynamics were validly executable. Neither ROLE-MRL nor ROLE-T produced a mechanically feasible candidate.

## Original B03 recheck

- Exact original candidates/orientations: 3 × 4 = 12; all original states had zero initial active storage contacts.
- Preload initializer: 0/12 feasible.
- Original retention: `{'10': 9, '25': 4, '50': 1, '100': 0, '200': 0, '500': 0, '1000': 0}`.
- Preloaded retention (frozen denominator 12): `{'10': 0, '25': 0, '50': 0, '100': 0, '200': 0, '500': 0, '1000': 0}`.
- Classification: `PR-E`; initialization did not materially alter the B03-C conclusion.

## Shape controls

- S0 sphere: 25 mm diameter; 12 frozen rechecks, 0 executable initializers.
- S1 cube: 25 mm side; 6 frozen candidates, 0 executable initializers.
- S2 short cylinder: 25 mm diameter × 20 mm height; 6 frozen candidates, 0 executable initializers. The 20 mm height is an engineering control.
- Shape effect is inconclusive because cube/cylinder retention dynamics could not validly start.

## Resource workspace

The geometric B03 arrangement retained thumb/index/opposition volumes of `{'thumb': 0.9559782183972225, 'index': 1.0, 'opposition': 0.9665998246424643}` relative to baseline. This is kinematic evidence only: the dynamically-supported workspace gate had zero eligible ≥200-step states. Object/storage-finger collision counts remain part of each descriptor.

## Role allocation

- ROLE-MRL: 1728 sampled, 329 prefilter passes, 6 frozen, 1 feasible initializer, 0 mechanically feasible, 0 robust.
- ROLE-T: 1728 sampled, 234 prefilter passes, 6 frozen, 1 feasible initializer, 0 mechanically feasible, 0 robust.

No morphology-specific role winner is supported. No handoff, object B, optimizer, RL, altered contact physics, or skin was used.
