from src.application.pipelines import BuildPsychologyMaster


def main():
    pipeline = BuildPsychologyMaster()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.master_repo.path}")


if __name__ == "__main__":
    main()
