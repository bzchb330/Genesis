#!/usr/bin/env python
from __future__ import annotations

from seqgrasp.experiments.phase2cm import Phase2CMStop, run_primary_phase2cm


def main() -> int:
    try:
        run_primary_phase2cm()
    except Phase2CMStop as exc:
        print(str(exc), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
