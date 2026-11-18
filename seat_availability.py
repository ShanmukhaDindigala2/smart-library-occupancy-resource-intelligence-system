
import pandas as pd


STUDY_AREAS = {
    "Reading Hall": {"capacity": 80, "weight": 1.00},
    "Silent Study Room": {"capacity": 30, "weight": 0.75},
    "Discussion Room": {"capacity": 20, "weight": 0.65},
    "Digital Study Area": {"capacity": 40, "weight": 0.85},
    "Reference Section": {"capacity": 30, "weight": 0.70},
    "Computer Lab": {"capacity": 25, "weight": 0.80},
    "Group Study Area": {"capacity": 35, "weight": 0.75},
    "Individual Study Cubicles": {"capacity": 20, "weight": 0.80},
    "Open Study Lounge": {"capacity": 25, "weight": 0.70},
}


def calculate_available_seats(total_seats, predicted_occupied):
    total_seats = max(0, int(total_seats))
    predicted_occupied = max(
        0.0,
        min(float(total_seats), float(predicted_occupied))
    )

    return int(round(total_seats - predicted_occupied))


def occupancy_status(predicted_occupied, total_seats):
    if total_seats <= 0:
        return "No Seats"

    predicted_occupied = max(
        0.0,
        min(float(total_seats), float(predicted_occupied))
    )

    percentage = (predicted_occupied / total_seats) * 100

    if percentage >= 85:
        return "Very High"
    elif percentage >= 70:
        return "High"
    elif percentage >= 40:
        return "Moderate"
    else:
        return "Low"


def predict_study_area_seats(occupancy_percentage):
    occupancy_percentage = max(
        0.0,
        min(100.0, float(occupancy_percentage))
    )

    rows = []

    for area, details in STUDY_AREAS.items():
        capacity = details["capacity"]
        weight = details["weight"]

        area_percentage = min(
            100.0,
            max(0.0, occupancy_percentage * weight)
        )

        filled = round(capacity * area_percentage / 100)
        empty = capacity - filled

        rows.append({
            "Study Area": area,
            "Total Seats": capacity,
            "Predicted Filled Seats": filled,
            "Predicted Empty Seats": empty,
            "Occupancy (%)": round(area_percentage, 1),
            "Status": occupancy_status(filled, capacity),
        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        by="Predicted Empty Seats",
        ascending=False
    ).reset_index(drop=True)


def get_recommended_study_area(occupancy_percentage):
    predictions = predict_study_area_seats(occupancy_percentage)

    available = predictions[
        predictions["Predicted Empty Seats"] > 0
    ]

    if available.empty:
        return "All study areas are predicted to be full."

    best = available.iloc[0]

    return (
        f"{best['Study Area']} - "
        f"{best['Predicted Empty Seats']} predicted empty seats"
    )


if __name__ == "__main__":
    total = 100
    occupied = 75

    print(
        "Available Seats:",
        calculate_available_seats(total, occupied)
    )

    print(
        "Occupancy Status:",
        occupancy_status(occupied, total)
    )

    print("\nStudy Area Predictions:")
    print(predict_study_area_seats(75).to_string(index=False))
