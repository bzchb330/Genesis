# Phase 3B-0.5 Orientation Analysis

Cumulative orientation change is descriptive, never a failure gate. The object
is a triaxial ellipsoid, so the symmetry-aware trace minimizes orientation
distance over the four D2 principal-axis symmetries. Angular speed, a 25-step
sustained angular-speed trace, support topology, gaps, and later retention are
reported alongside both rotation traces.

- total rotation distribution: `{"count": 240, "maximum": 3.141575505799311, "mean": 2.6543134128708963, "median": 2.729486281743166, "p90": 3.1410163725940294, "p95": 3.1414405753016865, "p99": 3.141543823505462}`
- D2 symmetry-aware rotation: `{"count": 240, "maximum": 2.050404634770468, "mean": 1.7102512846292783, "median": 1.6761280798720644, "p90": 1.921213400061428, "p95": 1.9402643099844652, "p99": 1.9556749521306214}`
- sustained angular speed: `{"count": 240, "maximum": 20.619691123702648, "mean": 16.138788657462804, "median": 16.432250065375488, "p90": 19.618772673967594, "p95": 20.070456289542413, "p99": 20.551118043212647}`
- final angular speed: `{"count": 240, "maximum": 2.1471118968004963, "mean": 0.050708539842925865, "median": 0.015770123877603626, "p90": 0.03680689017506023, "p95": 0.04480709365254897, "p99": 1.6307801728986395}`
- retained active trials: 101

Recommendation only: A5 should combine symmetry-aware orientation with
sustained angular speed and support/loss context. It should not use total
rotation alone; useful rolling or sliding must remain distinguishable from
uncontrolled rotation. No A5 threshold is frozen here.
