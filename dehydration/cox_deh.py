import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import plot_lifetimes
import matplotlib.pyplot as plt
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Generate mock dehydration dataset ---
def generate_mock_data(n=200, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        'time_hours': np.random.exponential(scale=10, size=n),  # Time until dehydration or censoring
        'dehydrated': np.random.binomial(1, p=0.7, size=n),
        'temperature': np.random.normal(loc=30, scale=4, size=n),  # in Celsius
        'activity_level': np.random.choice([1, 2, 3], size=n),     # 1=low, 2=med, 3=high
        'water_intake_liters': np.random.uniform(0.5, 3.5, size=n)
    })
    return data

# --- Load data and fit Cox model ---
def fit_model(data):
    cph = CoxPHFitter()
    cph.fit(data, duration_col='time_hours', event_col='dehydrated')
    cph.print_summary()
    return cph

# --- Plot Kaplan-Meier curve stratified by activity level ---
def plot_km_by_activity(data):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for level in sorted(data['activity_level'].unique()):
        mask = data['activity_level'] == level
        kmf.fit(data.loc[mask, 'time_hours'], data.loc[mask, 'dehydrated'], label=f'Activity {level}')
        kmf.plot_survival_function(ci_show=False)
    plt.title("Survival (No Dehydration) by Activity Level")
    plt.xlabel("Time (hours)")
    plt.ylabel("Probability of staying hydrated")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Predict median survival for a new profile ---
def predict_for_new_person(model, temperature, activity_level, water_intake_liters):
    df = pd.DataFrame([{
        'temperature': temperature,
        'activity_level': activity_level,
        'water_intake_liters': water_intake_liters
    }])
    prediction = model.predict_median(df)
    logging.info(f"Predicted median time until dehydration: {prediction.iloc[0]:.2f} hours")
    return prediction

# --- Main execution flow ---
if __name__ == "__main__":
    logging.info("Generating data...")
    df = generate_mock_data()

    logging.info("Fitting Cox model...")
    model = fit_model(df)

    logging.info("Plotting survival curves...")
    plot_km_by_activity(df)

    logging.info("Simulating prediction for new subject...")
    try:
        # Sample input for a test profile
        test_prediction = predict_for_new_person(
            model,
            temperature=35,
            activity_level=3,
            water_intake_liters=1.0
        )
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
