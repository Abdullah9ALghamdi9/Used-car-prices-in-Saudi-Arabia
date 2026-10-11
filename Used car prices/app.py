from fastapi import FastAPI
from catboost import CatBoostRegressor
import json
from pathlib import Path
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

app = FastAPI(title="Saudi Car Price Prediction API")

BASE_DIR = Path(__file__).resolve().parent

model = CatBoostRegressor()
model.load_model(str(BASE_DIR / "car_price_model.cbm"))

with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


@app.get("/")
def home():
    return {
        "message": "Saudi Car Price Prediction API is running",
        "model_loaded": True
    }

class CarInput(BaseModel):
    Make: str
    Type: str
    Year: int = Field(ge=1950, le=2030)
    Origin: str
    Options: str
    Engine_Size: float = Field(gt=0, le=20)
    Fuel_Type: str
    Gear_Type: str
    Mileage: float = Field(ge=0)
    Region: str


@app.post("/predict")
def predict_price(car: CarInput):
    data = car.model_dump()

    for col, allowed_values in metadata["rare_category_maps"].items():
        if data[col] not in allowed_values:
            data[col] = "Other"

    for col in metadata["categorical_features"]:
        if data[col] not in metadata["categorical_values"][col]:
            data[col] = "Other"

    df = pd.DataFrame([data])
    df = df[metadata["features"]]

    prediction_log = model.predict(df)[0]
    prediction = max(0, float(np.expm1(prediction_log)))

    return {
        "predicted_price": round(prediction, 2),
        "currency": "SAR"
    }
from fastapi.staticfiles import StaticFiles
app.mount('/ui', StaticFiles(directory=BASE_DIR / 'static', html=True), name='ui')


@app.get("/options")
def get_options():
    source = pd.read_csv(BASE_DIR / "UsedCarsSA_Unclean_EN.csv")
    source = source.loc[source["Price"] != "Negotiable"].copy()
    source["Price"] = pd.to_numeric(source["Price"], errors="coerce")
    source = source.dropna(subset=["Origin", "Options", "Gear_Type", "Price"])
    source = source.loc[(source["Price"] >= 5000) & (source["Mileage"] <= 1_000_000)].copy()
    for col, allowed in metadata["rare_category_maps"].items():
        source[col] = source[col].where(source[col].isin(allowed), "Other")

    # Use observed engine values only; do not invent engine sizes by imputation.
    engine_source = source.dropna(subset=["Engine_Size", "Make", "Type", "Year"]).copy()
    engine_source = engine_source.loc[engine_source["Engine_Size"].between(0.8, 9.0)]
    engine_source["Engine_Size"] = engine_source["Engine_Size"].round(1)

    # Remove clearly suspicious entries relative to each model's observed median.
    medians = engine_source.groupby(["Make", "Type"])["Engine_Size"].transform("median")
    tolerance = np.maximum(1.5, medians * 0.5)
    engine_source = engine_source.loc[(engine_source["Engine_Size"] - medians).abs() <= tolerance]

    engines = {}
    for (make, car_type, year), group in engine_source.groupby(["Make", "Type", "Year"]):
        engines.setdefault(str(make), {}).setdefault(str(car_type), {})[str(int(year))] = sorted(
            {float(v) for v in group["Engine_Size"]}
        )

    categorical = ["Origin", "Options", "Fuel_Type", "Gear_Type", "Region"]
    return {
        "makes_models": {
            str(make): sorted(map(str, group["Type"].dropna().unique()))
            for make, group in source.groupby("Make")
        },
        "years": sorted([int(v) for v in source["Year"].dropna().unique()], reverse=True),
        "engines_by_model_year": engines,
        "categories": {
            col: sorted(map(str, source[col].dropna().unique()))
            for col in categorical
        }
    }
