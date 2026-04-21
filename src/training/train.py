import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlflow.models.signature import infer_signature
import shap
import matplotlib.pyplot as plt

FEATURES = ["distance_km", "hour_of_day", "day_of_week",
            "is_weekend", "is_rush_hour", "is_night",
            "passenger_count", "PULocationID", "DOLocationID"]
TARGET = "trip_duration_seconds"

PARAMS = {
    "num_leaves": 64,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "random_state": 42,
}

def train(spark, feature_table, experiment_path, model_name):
    df = spark.read.format("delta").table(feature_table)
    pdf = df.drop("pickup_date").toPandas()

    X = pdf[FEATURES]
    y = pdf[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(experiment_path)

    with mlflow.start_run(run_name="lgbm-v1-baseline"):
        mlflow.log_params(PARAMS)
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        signature = infer_signature(X_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse_minutes", rmse / 60)

        mlflow.lightgbm.log_model(
            model, "lgbm_model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        # SHAP plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.sample(500, random_state=42))
        shap.summary_plot(shap_values, X_test.sample(500, random_state=42), show=False)
        plt.tight_layout()
        plt.savefig("/tmp/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("/tmp/shap_summary.png")

        print(f"RMSE: {rmse:.1f}s ({rmse/60:.1f} min)")
        print(f"MAE:  {mae:.1f}s ({mae/60:.1f} min)")