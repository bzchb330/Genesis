#!/usr/bin/env python
from __future__ import annotations

import json

from seqgrasp.experiments.phase2cm import run_contact_audit


def main() -> int:
    print(json.dumps(run_contact_audit(), indent=2, default=lambda value: value.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
