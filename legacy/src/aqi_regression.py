import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "results/aqi_regression/aqi_vit_dataset.npz"
OUTPUT_DIR = "results/aqi_regression"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AQI REGRESSION — CROSS-VALIDATED RIDGE")
print("=" * 70)

data = np.load(DATA_PATH)

X = data["embeddings"]
y = data["aqi"]
filenames = data["filenames"]
splits = data["splits"]

print("\nDataset:")
print("Embeddings:", X.shape)
print("AQI:", y.shape)

print("\nSamples by split:")
print(pd.Series(splits).value_counts())


# ============================================================
# ORIGINAL SPLIT
# ============================================================

train_mask = splits == "train"
val_mask = splits == "val"
test_mask = splits == "test"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

files_test = filenames[test_mask]

print("\nOriginal split:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []

for alpha in ALPHAS:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        return_train_score=False
    )

    mae_scores = -scores["test_mae"]
    rmse_scores = -scores["test_rmse"]
    r2_scores = scores["test_r2"]

    result = {
        "alpha": alpha,
        "cv_mae_mean": mae_scores.mean(),
        "cv_mae_std": mae_scores.std(),
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "cv_r2_mean": r2_scores.mean(),
        "cv_r2_std": r2_scores.std()
    }

    cv_results.append(result)

    print(f"\nalpha = {alpha}")

    print(
        f"MAE  : {result['cv_mae_mean']:.4f} "
        f"+/- {result['cv_mae_std']:.4f}"
    )

    print(
        f"RMSE : {result['cv_rmse_mean']:.4f} "
        f"+/- {result['cv_rmse_std']:.4f}"
    )

    print(
        f"R²   : {result['cv_r2_mean']:.4f} "
        f"+/- {result['cv_r2_std']:.4f}"
    )


# ============================================================
# SELECT BEST ALPHA
# ============================================================

cv_df = pd.DataFrame(cv_results)

best_row = cv_df.loc[
    cv_df["cv_mae_mean"].idxmin()
]

best_alpha = float(best_row["alpha"])

print("\n" + "=" * 70)
print("BEST RIDGE REGULARIZATION")
print("=" * 70)

print(f"Selected alpha: {best_alpha}")

print(
    f"CV MAE: {best_row['cv_mae_mean']:.4f}"
)

print(
    f"CV RMSE: {best_row['cv_rmse_mean']:.4f}"
)

print(
    f"CV R²: {best_row['cv_r2_mean']:.4f}"
)


# ============================================================
# SAVE CV RESULTS
# ============================================================

cv_path = os.path.join(
    OUTPUT_DIR,
    "ridge_cv_results.csv"
)

cv_df.to_csv(
    cv_path,
    index=False
)


# ============================================================
# FINAL TRAINING DATA
# ============================================================

# After selecting alpha using only the original training data,
# combine training + validation data.
#
# The test set remains untouched until final evaluation.

X_train_final = np.concatenate([
    X_train,
    X_val
])

y_train_final = np.concatenate([
    y_train,
    y_val
])

print("\n" + "=" * 70)
print("FINAL TRAINING")
print("=" * 70)

print(
    "Final training samples:",
    X_train_final.shape[0]
)

print(
    "Final test samples:",
    X_test.shape[0]
)


# ============================================================
# FINAL RIDGE MODEL
# ============================================================

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=best_alpha))
])

final_model.fit(
    X_train_final,
    y_train_final
)


# ============================================================
# SAVE TRAINED RIDGE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "aqi_ridge_model.joblib"
)

joblib.dump(
    final_model,
    model_path
)

print("\nSaved AQI model:")
print(model_path)


# ============================================================
# TEST PREDICTION
# ============================================================

y_test_pred = final_model.predict(X_test)


# ============================================================
# TEST METRICS
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# ============================================================
# MEAN-AQI BASELINE
# ============================================================

baseline_prediction = np.full(
    len(y_test),
    y_train_final.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print("\nViT Embedding + Ridge:")

print(
    f"MAE  : {test_mae:.4f}"
)

print(
    f"RMSE : {test_rmse:.4f}"
)

print(
    f"R²   : {test_r2:.4f}"
)

print("\nMean-AQI Baseline:")

print(
    f"MAE  : {baseline_mae:.4f}"
)

print(
    f"RMSE : {baseline_rmse:.4f}"
)

print(
    f"R²   : {baseline_r2:.4f}"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {
    "selected_alpha": best_alpha,

    "cross_validation": {
        "mae_mean": float(best_row["cv_mae_mean"]),
        "mae_std": float(best_row["cv_mae_std"]),
        "rmse_mean": float(best_row["cv_rmse_mean"]),
        "rmse_std": float(best_row["cv_rmse_std"]),
        "r2_mean": float(best_row["cv_r2_mean"]),
        "r2_std": float(best_row["cv_r2_std"])
    },

    "test_ridge": {
        "MAE": float(test_mae),
        "RMSE": float(test_rmse),
        "R2": float(test_r2)
    },

    "test_mean_baseline": {
        "MAE": float(baseline_mae),
        "RMSE": float(baseline_rmse),
        "R2": float(baseline_r2)
    },

    "dataset": {
        "total_samples": int(len(y)),
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "final_training_samples": int(len(y_train_final)),
        "embedding_dimensions": int(X.shape[1])
    },

    "model_file": model_path
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "aqi_regression_cv_metrics.json"
)

with open(metrics_path, "w") as f:
    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = pd.DataFrame({
    "filename": files_test,
    "actual_AQI": y_test,
    "predicted_AQI": y_test_pred,
    "absolute_error": np.abs(
        y_test - y_test_pred
    )
})

test_predictions_path = os.path.join(
    OUTPUT_DIR,
    "aqi_final_test_predictions.csv"
)

test_results.to_csv(
    test_predictions_path,
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_test_pred,
    s=80
)

min_value = min(
    y_test.min(),
    y_test_pred.min()
)

max_value = max(
    y_test.max(),
    y_test_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")

plt.title(
    "ViT Embeddings → AQI Prediction"
)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "aqi_final_actual_vs_predicted.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(cv_path)
print(metrics_path)
print(test_predictions_path)
print(plot_path)
print(model_path)

print("\n" + "=" * 70)
print("CROSS-VALIDATED AQI REGRESSION COMPLETE")
print("=" * 70)