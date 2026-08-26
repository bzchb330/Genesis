from __future__ import annotations

import json

from seqgrasp.phase3c06 import freeze_acquisition_states


if __name__ == "__main__":
    result = freeze_acquisition_states()
    print(json.dumps({
        "count": result["manifest"]["count"],
        "frozen_before_outcomes": result["manifest"]["frozen_before_outcomes"],
        "first_state": result["manifest"]["state_ids"][0],
        "last_state": result["manifest"]["state_ids"][-1],
    }, indent=2))
