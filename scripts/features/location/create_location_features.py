from src.application.pipelines import BuildLocationFeatures


def main():
    pipeline = BuildLocationFeatures()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.features_repo.path}")


if __name__ == "__main__":
    main()
