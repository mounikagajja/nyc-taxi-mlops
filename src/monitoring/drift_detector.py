from scipy import stats
import mlflow
import json

FEATURES = ["distance_km", "hour_of_day", "day_of_week",
            "is_weekend", "is_rush_hour", "is_night",
            "passenger_count", "trip_duration_seconds"]

DRIFT_THRESHOLD = 0.2

def calculate_drift(reference_df, current_df):
    drift_results = {}
    for col in FEATURES:
        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        drift_results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": bool(p_value < 0.05)
        }
    drifted = sum(1 for v in drift_results.values() if v["drifted"])
    drift_share = drifted / len(FEATURES)
    return drift_results, drift_share

def should_retrain(drift_share):
    if drift_share > DRIFT_THRESHOLD:
        print(f"ALERT: {drift_share:.0%} features drifted. Retraining recommended.")
        return True
    return False

def log_drift_to_mlflow(drift_results, drift_share):
    with mlflow.start_run(run_name="drift-monitor"):
        mlflow.log_metric("drift_share", drift_share)
        for col, result in drift_results.items():
            mlflow.log_metric(f"ks_{col}", result["ks_statistic"])
        with open("/tmp/drift_results.json", "w") as f:
            json.dump(drift_results, f, indent=2, default=str)
        mlflow.log_artifact("/tmp/drift_results.json")