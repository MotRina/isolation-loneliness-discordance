from src.application.pipelines import BuildLabelMaster


def main():
    pipeline = BuildLabelMaster()
    df = pipeline.run()
    print(df)
    print(f"Saved to: {pipeline.label_repo.path}")


if __name__ == "__main__":
    main()
