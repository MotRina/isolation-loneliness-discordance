from src.application.pipelines import BuildAnalysisMaster


def main():
    pipeline = BuildAnalysisMaster()
    df = pipeline.run()
    print(df.head())
    print(f"Saved to: {pipeline.master_repo.path}")


if __name__ == "__main__":
    main()
