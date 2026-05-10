from src.application.pipelines import BuildParticipantMapping


def main():
    pipeline = BuildParticipantMapping()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.mapping_repo.path}")


if __name__ == "__main__":
    main()
