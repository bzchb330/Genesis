from seqgrasp.phase3.audit import write_shadow_audit


if __name__ == "__main__":
    result = write_shadow_audit()
    print(result["compiled"])
