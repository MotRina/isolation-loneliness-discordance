from src.application.pipelines import BuildPhaseLocationFeatures


def main():
    pipeline = BuildPhaseLocationFeatures()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.features_repo.path}")


if __name__ == "__main__":
    main()
