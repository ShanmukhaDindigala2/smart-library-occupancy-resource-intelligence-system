import pandas as pd


def load_occupancy_data():
    df = pd.read_csv("data/occupancy_data.csv")

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.dropna()

    return df


def load_resource_data():
    df = pd.read_csv("data/resource_demand.csv")

    df["date"] = pd.to_datetime(df["date"])

    df = df.dropna()

    return df


def prepare_occupancy_features(df):

    features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "exam_period",
        "total_seats"
    ]

    X = df[features]
    y = df["occupied_seats"]

    return X, y


def prepare_demand_features(df):

    features = [
        "total_copies",
        "borrowed_copies",
        "exam_period"
    ]

    X = df[features]
    y = df["demand_count"]

    return X, y


if __name__ == "__main__":

    occupancy = load_occupancy_data()
    resource = load_resource_data()

    print("Occupancy Data:")
    print(occupancy.head())

    print("\nResource Data:")
    print(resource.head())