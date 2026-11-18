import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

# Load occupancy data
df = pd.read_csv("data/occupancy_data.csv")

# -----------------------------
# Average Occupancy by Hour
# -----------------------------

hourly = df.groupby("hour")["occupancy_rate"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly.index, hourly.values, marker="o")
plt.xlabel("Hour")
plt.ylabel("Average Occupancy (%)")
plt.title("Average Library Occupancy by Hour")
plt.grid(True)
plt.savefig("outputs/occupancy_plot.png")
plt.close()


# -----------------------------
# Exam vs Normal Occupancy
# -----------------------------

exam_data = df.groupby("exam_period")["occupancy_rate"].mean()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Normal Period", "Exam Period"],
    exam_data.values
)
plt.ylabel("Average Occupancy (%)")
plt.title("Occupancy During Exam and Normal Periods")
plt.savefig("outputs/exam_occupancy.png")
plt.close()


# -----------------------------
# Daily Occupancy
# -----------------------------

daily = df.groupby("date")["occupancy_rate"].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily.values)
plt.xlabel("Date")
plt.ylabel("Occupancy (%)")
plt.title("Daily Library Occupancy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/daily_occupancy.png")
plt.close()


print("Data analysis completed.")
print("Graphs saved inside outputs folder.")