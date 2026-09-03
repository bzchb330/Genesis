"""Run static audit only; B/C never launches preload or dynamic experiments."""
from seqgrasp.phase3cp05c import audit

if __name__=='__main__': audit()
