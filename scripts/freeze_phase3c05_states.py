from __future__ import annotations

import json

from seqgrasp.phase3c05 import freeze_matched_states


if __name__ == "__main__":
    result = freeze_matched_states()
    print(json.dumps({
        "count": result["manifest"]["count"],
        "state_ids": [state.state_id for state in result["states"]],
        "candidate_ids": [state.source_candidate for state in result["states"]],
        "storage_entry_steps": [state.storage_entry_step for state in result["states"]],
    }, indent=2))
