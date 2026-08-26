from __future__ import annotations

import json

from seqgrasp.phase3c0_experiments import run_phase3c0


if __name__ == "__main__":
    result = run_phase3c0()
    print(json.dumps({
        "phase": result["phase"],
        "summary": result["C0_A_and_B"]["summary"],
        "first_object_validation_gate": result["first_object_validation_gate"],
        "wrist": {key: value for key, value in result["C0_C_wrist_feasibility"].items()
                  if key != "candidates"},
        "insertion": result["C0_E_F_B_insertion_and_resecure"],
    }, indent=2))
