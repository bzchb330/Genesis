from __future__ import annotations
import json
from seqgrasp.phase3c11 import freeze_preloaded_b03_manifest,run_preloaded_b03_holds

if __name__=="__main__":
    manifest=freeze_preloaded_b03_manifest(); print(json.dumps({"manifest_sha256":manifest["sha256"],"feasible":sum(r["initializer_feasible"] for r in manifest["rows"])},indent=2))
    result=run_preloaded_b03_holds(); print(json.dumps({"classification":result["classification"],"counts":result["preloaded_survival_counts"],"feasible_initializers":result["feasible_initializers"]},indent=2))
