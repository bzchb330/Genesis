"""Explicit P0 stages. Never runs a receiver or bypasses PI physics approval."""
import argparse
import json
from seqgrasp import phase3cp0 as p

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['audit','legacy','candidates','regression','hand-gate'])
args=parser.parse_args()
if args.stage=='audit': p.run_audit()
elif args.stage=='legacy': p.run_suite(p.config()['legacy_name'])
elif args.stage=='candidates':
    for option in p.frozen_config()['candidate_options']: p.run_suite(option['name'],option)
elif args.stage=='regression': print(json.dumps(p.legacy_regression(),indent=2))
elif args.stage=='hand-gate':
    p.require_validated_physics(p.read('legacy_classification.json')['classification'],p.config()['approved_revised_physics'])
    raise NotImplementedError('PI-approved hand control is a separate gated stage, not an automatic benchmark continuation')
