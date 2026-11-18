import pandas as pd


def detect_peak_hours():

    df = pd.read_csv("data/occupancy_data.csv")

    hourly = (
        df.groupby("hour")["occupancy_rate"]
        .mean()
        .sort_values(ascending=False)
    )

    peak_hours = hourly.head(5)

    return peak_hours


if __name__ == "__main__":

    peak_hours = detect_peak_hours()

    print("Peak Library Hours:")
    print(peak_hours)