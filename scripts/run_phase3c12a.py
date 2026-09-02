"""Frozen Phase 3C-1.2A entrypoint; no receiver hold dynamics."""
from __future__ import annotations
import argparse
from seqgrasp.phase3c12a import (old_calibration_autopsy, corrected_calibration, run_mechanics,
                               complete_offline_audits, audit_frozen_local_candidates)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--part',choices=['calibration','mechanics','all'],default='all'); args=parser.parse_args()
    if args.part in ('calibration','all'): old_calibration_autopsy(); corrected_calibration()
    if args.part in ('mechanics','all'):
        run_mechanics(); complete_offline_audits(); audit_frozen_local_candidates()
