# Saudi Used Car Price Prediction 

A machine learning project that predicts used car prices in Saudi Arabia based on vehicle specifications such as make, model, year, mileage, and engine size.

The project includes data cleaning, preprocessing, model training, evaluation, and a web interface connected to the trained model through FastAPI.

### Model Performance

The model was trained using **CatBoost Regressor**.

- **R² Score:** 0.844
- **MAE:** 13,316 SAR
- **Cross-Validation R²:** 0.828

### Technologies Used

Python, Pandas, NumPy, Scikit-learn, CatBoost, FastAPI, Jupyter Notebook.

### How It Works

Users select a vehicle's specifications, and the model predicts its estimated price in Saudi Riyals.

### Run Locally

Install the required packages:

```bash
pip install fastapi "uvicorn[standard]" catboost pandas numpy scikit-learn
```

Run the application:

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/ui/` in your browser.
