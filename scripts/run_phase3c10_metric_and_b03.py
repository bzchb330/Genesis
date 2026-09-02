from __future__ import annotations
import json
from seqgrasp.phase3c10 import OUTPUT, freeze_b03_manifest, metric_repair_audit, run_b03_validation

if __name__=="__main__":
    OUTPUT.mkdir(parents=True,exist_ok=True); metric=metric_repair_audit(); (OUTPUT/"metric_repair_audit.json").write_text(json.dumps(metric,indent=2),encoding="utf-8"); print(json.dumps(metric,indent=2))
    manifest=freeze_b03_manifest(); print(json.dumps({"manifest_sha256":manifest["sha256"],"trial_count":manifest["trial_count"],"candidate_ids":[c["candidate_id"] for c in manifest["candidates"]["selected"]],"orientations":[o["orientation_id"] for o in manifest["orientations"]]},indent=2))
    result=run_b03_validation(); print(json.dumps({"classification":result["classification"],"approved":result["approved_as_transport_target"],"survival_counts":result["survival_counts"]},indent=2))
