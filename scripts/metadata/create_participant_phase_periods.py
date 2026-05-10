from src.application.pipelines import BuildParticipantPhasePeriods


def main():
    pipeline = BuildParticipantPhasePeriods()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.periods_repo.path}")


if __name__ == "__main__":
    main()
