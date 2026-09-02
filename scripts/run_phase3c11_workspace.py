from __future__ import annotations
import json
from seqgrasp.phase3c11 import resource_workspace_audit
if __name__=="__main__":
    result=resource_workspace_audit(); print(json.dumps({"baseline":{f:result["baseline"][f]["reachable_volume_m3"] for f in ("thumb","index")},"geometric_B03":{f:result["geometric_B03"][f]["reachable_volume_m3"] for f in ("thumb","index")},"retained_fraction":result["retained_fraction"],"dynamic_gate":result["dynamic_workspace_gate"]},indent=2))
