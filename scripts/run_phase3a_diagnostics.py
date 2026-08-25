from seqgrasp.phase3.experiments import write_phase3a_results


if __name__ == "__main__":
    cohort, handoff = write_phase3a_results()
    print(
        {
            "minimal_attempts": cohort["minimal_attempts"],
            "minimal_classification_counts": cohort["minimal_classification_counts"],
            "middle_recruitment_attempts": cohort["middle_recruitment_attempts"],
            "handoff": handoff["summary"],
        }
    )
