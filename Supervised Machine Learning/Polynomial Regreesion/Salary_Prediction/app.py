import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="INR Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Salary Predictor (Polynomial + Department)")
st.write("Predict `Monthly_Salary_INR` from `Employee_ID` and `Department` using `Salary_Data.csv`.")

DATA_PATH = os.path.join("dataset", "Salary_Data.csv")
MODEL_PATH = os.path.join("model", "salary_poly_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("Model artifact not found. Run `python train.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

st.sidebar.header("Input Parameters")
departments = sorted(df["Department"].unique().tolist())
dept_input = st.sidebar.selectbox("Department", departments)
id_input = st.sidebar.slider(
    "Employee_ID",
    min_value=int(df["Employee_ID"].min()),
    max_value=int(df["Employee_ID"].max()),
    value=50,
    step=1,
)

input_df = pd.DataFrame({"Employee_ID": [id_input], "Department": [dept_input]})
predicted_inr = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.metric(
    label=f"Estimated Monthly Salary ({dept_input}, ID {id_input})",
    value=f"₹{predicted_inr:,.2f}",
)

st.subheader("Curve for Selected Department")
grid_ids = np.linspace(df["Employee_ID"].min(), df["Employee_ID"].max(), 200)
grid_df = pd.DataFrame({"Employee_ID": grid_ids, "Department": [dept_input] * len(grid_ids)})
grid_preds = model.predict(grid_df)

dept_rows = df[df["Department"] == dept_input]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(dept_rows["Employee_ID"], dept_rows["Monthly_Salary_INR"], color="#e63946", label=f"{dept_input} Actuals", zorder=3)
ax.plot(grid_ids, grid_preds, color="#1d3557", linewidth=2, label=f"Model Curve ({dept_input})")
ax.scatter([id_input], [predicted_inr], color="#2a9d8f", s=120, marker="D", label="Prediction", zorder=4)
ax.set_title(f"Employee_ID vs. Monthly_Salary_INR ({dept_input})")
ax.set_xlabel("Employee_ID")
ax.set_ylabel("Monthly Salary (INR)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig)

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)