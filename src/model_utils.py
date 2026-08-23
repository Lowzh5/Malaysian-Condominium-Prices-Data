import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone


def evaluate_model(model, X, y_log, label=""):
    """Evaluate a model trained on log(price) against log(price) targets.
    Predictions and actuals are converted back to RM (np.exp) before computing
    metrics, since price was log-transformed in Section 3.6 but the report
    defines RMSE/MAE/MSE in RM (Section 1.8).

    Also reports MAPE (the % version of MAE, averaged per-listing) and RMSE
    as a % of the median actual price (RMSE_pct, using median rather than
    mean to stay consistent with the report's median-based treatment of the
    right-skewed price distribution) - both make the raw RM error sizes
    easier to judge without picking an arbitrary "good/bad" threshold."""
    y_pred_log = model.predict(X)
    y_true_rm = np.exp(y_log)
    y_pred_rm = np.exp(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true_rm, y_pred_rm))
    mae = mean_absolute_error(y_true_rm, y_pred_rm)
    r2 = r2_score(y_true_rm, y_pred_rm)
    mse = mean_squared_error(y_true_rm, y_pred_rm)
    mape = np.mean(np.abs((y_true_rm - y_pred_rm) / y_true_rm)) * 100
    rmse_pct = rmse / np.median(y_true_rm) * 100

    if label:
        print(f"{label} RMSE:  RM {rmse:,.0f}  ({rmse_pct:.1f}% of median price)")
        print(f"{label} MAE:   RM {mae:,.0f}")
        print(f"{label} MAPE:  {mape:.1f}%")
        print(f"{label} R2:    {r2:.4f}")
        print(f"{label} MSE:   {mse:,.0f}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MSE": mse,
            "MAPE": mape, "RMSE_pct": rmse_pct}


def cross_validate_model(model, X, y_log, n_splits=5, random_state=42):
    """5-fold CV on the training set only (X_test stays untouched for final
    evaluation). Each fold re-fits a fresh clone of the model and scores it
    with evaluate_model, so the RM-scale conversion and % metrics stay
    consistent with the train/test numbers reported elsewhere."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = X.reset_index(drop=True)
    y_log = y_log.reset_index(drop=True)

    fold_results = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        fold_results.append(evaluate_model(fold_model, X_val, y_val))

    keys = fold_results[0].keys()
    mean = {k: np.mean([f[k] for f in fold_results]) for k in keys}
    std = {k: np.std([f[k] for f in fold_results]) for k in keys}

    print(f"{n_splits}-fold CV (mean +/- std):")
    print(f"  RMSE:  RM {mean['RMSE']:,.0f} +/- {std['RMSE']:,.0f}  ({mean['RMSE_pct']:.1f}% of median price)")
    print(f"  MAE:   RM {mean['MAE']:,.0f} +/- {std['MAE']:,.0f}")
    print(f"  MAPE:  {mean['MAPE']:.1f}% +/- {std['MAPE']:.1f}%")
    print(f"  R2:    {mean['R2']:.4f} +/- {std['R2']:.4f}")
    print(f"  MSE:   {mean['MSE']:,.0f} +/- {std['MSE']:,.0f}")

    return {"fold_results": fold_results, "mean": mean, "std": std}
