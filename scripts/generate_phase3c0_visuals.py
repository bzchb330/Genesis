from __future__ import annotations

import json

from seqgrasp.phase3c0_visuals import generate_phase3c0_visuals


if __name__ == "__main__":
    print(json.dumps(generate_phase3c0_visuals(), indent=2))
