import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


def train_model(degree: int = 4):
    data_path = os.path.join("dataset", "Salary_Data.csv")
    model_dir = "model"
    model_path = os.path.join(model_dir, "salary_poly_model.pkl")

    df = pd.read_csv(data_path)
    X = df[["Employee_ID", "Department"]]
    y = df["Monthly_Salary_INR"].values

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "poly_id",
                Pipeline([
                    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ]),
                ["Employee_ID"],
            ),
            (
                "dept",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                ["Department"],
            ),
        ]
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    model_pipeline.fit(X, y)
    preds = model_pipeline.predict(X)

    print(f"Trained Polynomial + Department Pipeline (Degree={degree})")
    print(f"R² Score: {r2_score(y, preds):.4f}")
    print(f"RMSE:     ₹{np.sqrt(mean_squared_error(y, preds)):,.2f}")
    print(f"MAE:      ₹{mean_absolute_error(y, preds):,.2f}")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model_pipeline, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_model(degree=4)