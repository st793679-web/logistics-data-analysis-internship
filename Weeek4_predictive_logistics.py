import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import docx
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Dataset Simulation
np.random.seed(42)
n_samples = 5000

distance = np.random.uniform(50, 1500, n_samples)
volume = np.random.uniform(1, 50, n_samples)
traffic = np.random.uniform(1.0, 5.0, n_samples)
weather = np.random.uniform(1.0, 5.0, n_samples)
carrier_tier = np.random.choice(
    ["Tier 1", "Tier 2", "Tier 3"], size=n_samples, p=[0.4, 0.4, 0.2]
)
transport_mode = np.random.choice(
    ["Road", "Air", "Rail", "Sea"], size=n_samples, p=[0.5, 0.2, 0.2, 0.1]
)

# Synthetic Target with Non-linear Dependencies & Noise
delivery_time = (
    (distance / 60.0)
    + (volume * 0.15)
    + (traffic * 2.5)
    + (weather * 3.1)
    + np.where(carrier_tier == "Tier 3", 6.0, 0.0)
    + np.random.normal(0, 2.0, n_samples)
)

df = pd.DataFrame({
    "distance_km": distance,
    "cargo_volume_m3": volume,
    "traffic_index": traffic,
    "weather_index": weather,
    "carrier_tier": carrier_tier,
    "transport_mode": transport_mode,
    "delivery_time_hours": delivery_time,
})

# 2. Predictive Pipeline Setup
X = df.drop(columns=["delivery_time_hours"])
y = df["delivery_time_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_cols = ["carrier_tier", "transport_mode"]
num_cols = ["distance_km", "cargo_volume_m3", "traffic_index", "weather_index"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop="first"), cat_cols),
    ("num", "passthrough", num_cols),
])

model = XGBRegressor(
    n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42
)

model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", model)]
)

# 3. Model Training & Metrics Output
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(
    f"Model Evaluated successfully:\nMAE: {mae:.2f} hrs | RMSE: {rmse:.2f} hrs"
    f" | R²: {r2:.3f}"
)