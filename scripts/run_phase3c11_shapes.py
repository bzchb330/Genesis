from __future__ import annotations
import json
from seqgrasp.phase3c11 import freeze_shape_candidates,run_shape_holds
if __name__=="__main__":
    manifest=freeze_shape_candidates(); print(json.dumps({k:{"prefilter":v["prefilter_passed"],"selected":len(v["selected"]),"feasible":sum(r["initializer_feasible"] for r in v["initialized"])} for k,v in manifest["shapes"].items()},indent=2))
    result=run_shape_holds(); print(json.dumps({k:v["survival_counts"] for k,v in result["shapes"].items()},indent=2))
