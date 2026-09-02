from __future__ import annotations
import json
from seqgrasp.phase3c11 import write_contact_and_calibration_audit

if __name__ == "__main__":
    value=write_contact_and_calibration_audit()
    print(json.dumps({"surfaces":list(value["calibration"]["selection"]["targets"]),"selection":value["calibration"]["selection"],"rows":len(value["calibration"]["rows"])},indent=2))
