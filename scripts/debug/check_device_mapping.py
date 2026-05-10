from src.infrastructure.database import DeviceRepository


def main():
    repo = DeviceRepository()
    df = repo.fetch_sample(limit=20)
    print(df.to_string())


if __name__ == "__main__":
    main()
