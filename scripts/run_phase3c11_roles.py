from __future__ import annotations
import json
from seqgrasp.phase3c11 import storage_role_mechanics_search,run_role_T_holds
if __name__=="__main__":
    result=storage_role_mechanics_search(); print(json.dumps({name:{"search":role["search_size"],"prefilter":role["prefilter_count"],"selected":len(role["selected"]),"mechanically_feasible":role["mechanically_feasible_count"],"disturbance_robust":role["disturbance_robust_count"]} for name,role in result["roles"].items()},indent=2)); print(json.dumps(run_role_T_holds(),indent=2))
