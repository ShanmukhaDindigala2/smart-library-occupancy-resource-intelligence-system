import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/resource_demand.csv")

features = [
    "total_copies",
    "borrowed_copies",
    "exam_period"
]

X = df[features]
y = df["demand_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

joblib.dump(model, "models/demand_model.pkl")

print("Demand model trained successfully.")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))
print("Saved as models/demand_model.pkl")