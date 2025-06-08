import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any
import sys

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dehydration_analysis.log")
    ]
)

def generate_mock_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dehydration data with realistic covariate relationships.
    
    Parameters:
        n: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: time_hours, dehydrated, temperature, 
        activity_level, water_intake_liters
    """
    np.random.seed(seed)
    
    # Generate covariates with realistic distributions
    temperature = np.random.normal(loc=30, scale=4, size=n)
    activity_level = np.random.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2])
    water_intake = np.clip(np.random.normal(loc=2.0, scale=0.8, size=n), 0.5, 3.5)
    
    # Create realistic hazard relationships
    hazard = (
        0.05 + 
        0.025 * (temperature - 30) +  # Higher temp → higher risk
        0.15 * (activity_level - 1) -  # Higher activity → higher risk
        0.25 * (water_intake - 2.0)    # More water → lower risk
    )
    
    # Generate survival times based on hazards
    survival_times = np.random.exponential(scale=1/np.exp(hazard), size=n)
    
    # Generate censoring (about 30% of cases)
    censored = np.random.binomial(1, p=0.3, size=n)
    
    data = pd.DataFrame({
        'time_hours': np.clip(survival_times, 0, 100),  # Cap at 100 hours
        'dehydrated': 1 - censored,  # 1 = event observed, 0 = censored
        'temperature': temperature,
        'activity_level': activity_level,
        'water_intake_liters': water_intake
    })
    
    # Convert to categorical after DataFrame creation
    data['activity_level'] = data['activity_level'].astype('category')
    
    return data

def fit_model(data: pd.DataFrame) -> CoxPHFitter:
    """
    Fit and validate Cox Proportional Hazards model.
    
    Parameters:
        data: DataFrame containing survival data
        
    Returns:
        Fitted CoxPHFitter model
    """
    cph = CoxPHFitter(penalizer=0.1)  # Small L2 penalty for stability
    cph.fit(data, duration_col='time_hours', event_col='dehydrated')
    
    # Model diagnostics
    logging.info("\n=== Model Summary ===")
    cph.print_summary()
    
    logging.info("\n=== Proportional Hazards Test ===")
    ph_test = proportional_hazard_test(cph, data, time_transform='rank')
    print(ph_test.print_summary(decimals=3))
    
    logging.info(f"\nConcordance Index: {cph.concordance_index_:.3f}")
    
    return cph

def plot_km_by_activity(data: pd.DataFrame) -> None:
    """
    Plot stratified Kaplan-Meier curves with confidence intervals.
    
    Parameters:
        data: DataFrame containing survival data
    """
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 7))
    
    # Create plot with at-risk counts
    ax = plt.subplot(111)
    for level in sorted(data['activity_level'].unique()):
        mask = data['activity_level'] == level
        kmf.fit(
            data.loc[mask, 'time_hours'],
            data.loc[mask, 'dehydrated'],
            label=f'Activity Level {level}'
        )
        kmf.plot_survival_function(ax=ax, ci_show=True)
    
    # Add at-risk table
    add_at_risk_counts(kmf, ax=ax)
    
    plt.title("Survival (No Dehydration) by Activity Level", pad=20)
    plt.xlabel("Time (hours)")
    plt.ylabel("Probability of Staying Hydrated")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

def predict_for_new_person(
    model: CoxPHFitter,
    temperature: float,
    activity_level: int,
    water_intake_liters: float,
    percentiles: list = [0.25, 0.5, 0.75]
) -> Dict[str, Any]:
    """
    Predict survival characteristics for a new individual.
    
    Parameters:
        model: Fitted CoxPHFitter model
        temperature: Temperature in Celsius
        activity_level: 1=low, 2=med, 3=high
        water_intake_liters: Daily water intake in liters
        percentiles: Survival percentiles to calculate
        
    Returns:
        Dictionary containing prediction results with formatted strings
    """
    # Create input DataFrame with proper categorical encoding
    df = pd.DataFrame({
        'temperature': [temperature],
        'activity_level': pd.Categorical([activity_level], categories=[1, 2, 3]),
        'water_intake_liters': [water_intake_liters]
    })
    
    try:
        # Calculate predictions
        median_pred = float(model.predict_median(df))
        percentile_preds = {
            f'{p*100}%': float(model.predict_percentile(df, p=p))
            for p in percentiles
        }
        hazard_pred = float(model.predict_partial_hazard(df))
        
        # Format results for clear interpretation
        predictions = {
            'median_survival': "Not reached" if np.isinf(median_pred) else f"{median_pred:.2f} hours",
            'percentiles': {
                p: "Not reached" if np.isinf(v) else f"{v:.2f} hours"
                for p, v in percentile_preds.items()
            },
            'hazard_ratio': f"{hazard_pred:.4f}"
        }
        
        # Log formatted results
        logging.info("\n=== Prediction Results ===")
        logging.info(f"Temperature: {temperature}°C")
        logging.info(f"Activity Level: {activity_level}")
        logging.info(f"Water Intake: {water_intake_liters} L/day")
        
        for k, v in predictions.items():
            if isinstance(v, dict):
                logging.info(f"\n{k.title()}:")
                for p, val in v.items():
                    logging.info(f"  {p}: {val}")
            else:
                logging.info(f"{k.replace('_', ' ').title()}: {v}")
        
        return predictions
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    try:
        logging.info("=== Starting Dehydration Survival Analysis ===")
        
        # Generate and inspect data
        logging.info("\nGenerating realistic mock data...")
        df = generate_mock_data(n=300)
        
        logging.info("\n=== Data Summary ===")
        logging.info(f"Total samples: {len(df)}")
        logging.info(f"Dehydration events: {df['dehydrated'].sum()}")
        logging.info(f"Censored observations: {len(df) - df['dehydrated'].sum()}")
        logging.info("\nFirst 3 rows:")
        logging.info(df.head(3).to_string())
        
        # Fit model
        logging.info("\nFitting Cox proportional hazards model...")
        model = fit_model(df)
        
        # Visualize results
        logging.info("\nGenerating survival plots...")
        plot_km_by_activity(df)
        
        # Make predictions for test cases
        logging.info("\n=== Making Predictions for Test Profiles ===")
        test_profiles = [
            {"temp": 25, "activity": 1, "water": 3.0, "desc": "Low risk (cool, sedentary, well-hydrated)"},
            {"temp": 30, "activity": 2, "water": 2.0, "desc": "Moderate risk"},
            {"temp": 35, "activity": 3, "water": 1.0, "desc": "High risk (hot, active, low water intake)"},
            {"temp": 40, "activity": 3, "water": 0.5, "desc": "Extreme risk case"}
        ]
        
        for profile in test_profiles:
            logging.info(f"\n* Profile: {profile['desc']}")
            predictions = predict_for_new_person(
                model,
                temperature=profile["temp"],
                activity_level=profile["activity"],
                water_intake_liters=profile["water"]
            )
            
        logging.info("\n=== Analysis Completed Successfully ===")
        
    except Exception as e:
        logging.error(f"\n!!! Analysis Failed: {str(e)}", exc_info=True)
        sys.exit(1)
