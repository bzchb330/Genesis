"""Natural hand-only settling and settled sphere geometry, no contact comparison."""
from seqgrasp.phase3cp05r import prepare_equilibrium, prepare_geometry

if __name__=='__main__':
    prepare_equilibrium()
    print('GEOMETRY',prepare_geometry()['valid'])
