# model_utils.py
# Handles preprocessing, training three ML models, evaluating them,
# and making new predictions.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import get_categorical_and_numerical_cols


# ─────────────────────────────────────────────
#  Build a preprocessing + model Pipeline
# ─────────────────────────────────────────────

def build_pipeline(model, cat_cols, num_cols, scale_numeric=False):
    """
    Create a scikit-learn Pipeline that:
      1. One-hot encodes categorical columns.
      2. Optionally scales numerical columns (for Linear Regression).
         Leaves numerical columns unscaled for tree-based models.
      3. Fits the given regression model.

    Parameters
    ----------
    model         : a scikit-learn estimator
    cat_cols      : list of categorical column names
    num_cols      : list of numerical column names
    scale_numeric : bool, True to apply StandardScaler (useful for Linear Regression)

    Returns
    -------
    sklearn Pipeline
    """

    num_transformer = StandardScaler() if scale_numeric else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    model),
    ])

    return pipeline


# ─────────────────────────────────────────────
#  Train all three models
# ─────────────────────────────────────────────

def train_models(X, y):
    """
    Split data, build three pipelines, train them, and return everything
    needed for evaluation and prediction.
    """

    # 80 % training, 20 % testing  |  random_state ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_cols, num_cols = get_categorical_and_numerical_cols(X_train)

    # ── Define models and whether numerical scaling is useful ──
    model_configs = {
        # Linear Regression benefits from standard scaling of numerical features
        "Linear Regression": (LinearRegression(), True),

        # Tree-based models do not require feature scaling
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False),
    }

    pipelines = {}
    for name, (model, scale_num) in model_configs.items():
        pipe = build_pipeline(model, cat_cols, num_cols, scale_numeric=scale_num)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

    return {
        "X_train":   X_train,
        "X_test":    X_test,
        "y_train":   y_train,
        "y_test":    y_test,
        "pipelines": pipelines,
        "cat_cols":  cat_cols,
        "num_cols":  num_cols,
    }


# ─────────────────────────────────────────────
#  Evaluate all trained models
# ─────────────────────────────────────────────

def evaluate_models(trained):
    """
    Compute MAE, RMSE, and R² for each trained pipeline on the test set.

    Returns a DataFrame with one row per model.
    """
    X_test  = trained["X_test"]
    y_test  = trained["y_test"]
    results = []

    for name, pipe in trained["pipelines"].items():
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE (₹)":  round(mae, 2),
            "RMSE (₹)": round(rmse, 2),
            "R² Score": round(r2, 4),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  Predict a single car's price
# ─────────────────────────────────────────────

def predict_price(pipeline, input_dict):
    """
    Predict the selling price for one car.

    Parameters
    ----------
    pipeline   : a trained sklearn Pipeline
    input_dict : dict of feature_name → value  (matches training columns)

    Returns
    -------
    float – predicted price in INR
    """
    input_df = pd.DataFrame([input_dict])
    prediction = pipeline.predict(input_df)[0]
    return max(prediction, 0)  # price cannot be negative


# ─────────────────────────────────────────────
#  Feature importance helper
# ─────────────────────────────────────────────

def get_feature_importance(pipeline, cat_cols, num_cols):
    """
    Extract feature importances from a tree-based model inside a Pipeline.

    Returns a DataFrame with columns ['Feature', 'Importance'] sorted
    by importance descending, or None if the model doesn't support it.
    """
    regressor = pipeline.named_steps["regressor"]

    if not hasattr(regressor, "feature_importances_"):
        return None

    # Recover feature names after one-hot encoding
    preprocessor = pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"]
    encoded_cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = encoded_cat_names + num_cols

    importances = regressor.feature_importances_

    feat_df = pd.DataFrame({
        "Feature":    all_feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return feat_df


# ─────────────────────────────────────────────
#  Actual vs Predicted helper
# ─────────────────────────────────────────────

def get_actual_vs_predicted(trained, model_name):
    """
    Return a DataFrame with 'Actual' and 'Predicted' columns for the
    chosen model, sampled to at most 300 rows for readability.
    """
    pipe   = trained["pipelines"][model_name]
    X_test = trained["X_test"]
    y_test = trained["y_test"]

    y_pred = pipe.predict(X_test)

    av_df = pd.DataFrame({
        "Actual":    y_test.values,
        "Predicted": y_pred,
    })

    # Sample for chart performance
    if len(av_df) > 300:
        av_df = av_df.sample(300, random_state=42).reset_index(drop=True)

    return av_df
