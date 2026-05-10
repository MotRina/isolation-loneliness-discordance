from src.application.pipelines import BuildQuestionnaireMaster


def main():
    pipeline = BuildQuestionnaireMaster()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.master_repo.path}")


if __name__ == "__main__":
    main()
