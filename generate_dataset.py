
import os
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

RESOURCES = {
    "Programming Books": 80,
    "AI & ML Books": 60,
    "Data Structures Books": 75,
    "Database Books": 50,
    "Research Papers": 40,
    "Digital Resources": 100,
    "Reference Books": 70
}


def generate_occupancy_data():
    dates = pd.date_range("2026-01-01", "2026-06-30", freq="D")
    records = []

    for date in dates:
        day = date.dayofweek
        weekend = int(day >= 5)
        exam = int(random.random() < 0.25)

        for hour in range(24):
            if hour < 6:
                factor = 0.08
            elif hour < 9:
                factor = 0.25 + (hour - 6) * 0.12
            elif hour < 14:
                factor = 0.65
            elif hour < 16:
                factor = 0.45
            elif hour < 20:
                factor = 0.80
            elif hour < 22:
                factor = 0.45
            else:
                factor = 0.15

            if weekend:
                factor *= 0.70

            if exam:
                factor += 0.15

            total_seats = 100
            noise = np.random.normal(0, 5)
            occupied = int(round(
                total_seats * min(0.98, max(0.02, factor)) + noise
            ))
            occupied = max(0, min(total_seats, occupied))

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "hour": hour,
                "day_of_week": day,
                "is_weekend": weekend,
                "exam_period": exam,
                "total_seats": total_seats,
                "occupied_seats": occupied,
                "occupancy_rate": round(occupied / total_seats * 100, 2)
            })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, "occupancy_data.csv")
    df.to_csv(path, index=False)

    print("Occupancy dataset created:", path)
    print("Rows:", len(df))


def generate_resource_demand_data():
    dates = pd.date_range("2026-01-01", "2026-06-30", freq="D")
    records = []

    for date in dates:
        day = date.dayofweek
        weekend = int(day >= 5)
        exam = int(random.random() < 0.25)

        for hour in range(24):
            if hour < 6:
                activity = 0.10
            elif hour < 9:
                activity = 0.40
            elif hour < 14:
                activity = 0.75
            elif hour < 16:
                activity = 0.50
            elif hour < 20:
                activity = 0.90
            elif hour < 22:
                activity = 0.45
            else:
                activity = 0.15

            if weekend:
                activity *= 0.75

            for resource, total_copies in RESOURCES.items():
                exam_boost = 0.20 if exam else 0
                resource_factor = (
                    1.2 if resource in [
                        "Programming Books",
                        "AI & ML Books",
                        "Data Structures Books",
                        "Digital Resources"
                    ] else 1.0
                )

                borrowed = int(round(
                    total_copies * min(
                        0.90,
                        max(0.02, activity * resource_factor + exam_boost)
                    )
                ))
                borrowed = min(total_copies, max(0, borrowed))

                demand = max(
                    1,
                    int(round(
                        borrowed * 0.65
                        + (15 if exam else 0)
                        + np.random.normal(0, 3)
                    ))
                )

                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": day,
                    "is_weekend": weekend,
                    "exam_period": exam,
                    "resource_name": resource,
                    "total_copies": total_copies,
                    "borrowed_copies": borrowed,
                    "demand_count": demand
                })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, "resource_demand.csv")
    df.to_csv(path, index=False)

    print("Resource demand dataset created:", path)
    print("Rows:", len(df))


if __name__ == "__main__":
    print("Generating 24-hour Smart Library datasets...")
    generate_occupancy_data()
    generate_resource_demand_data()
    print("Both datasets generated successfully.")
