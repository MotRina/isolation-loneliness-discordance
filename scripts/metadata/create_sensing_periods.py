from src.application.pipelines import BuildParticipantSensingPeriods


def main():
    pipeline = BuildParticipantSensingPeriods()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.periods_repo.path}")


if __name__ == "__main__":
    main()
