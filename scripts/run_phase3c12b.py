"""Explicit stages preserve the frozen experimental execution order."""
import argparse
from seqgrasp import phase3c12b as phase

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['audit','primitives','construction','release'])
    args=parser.parse_args()
    {'audit':phase.actuation_audit,'primitives':phase.run_primitives,
     'construction':phase.run_construction,'release':phase.run_release}[args.stage]()
