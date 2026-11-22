from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Salary_Data.csv"
MODEL_PATH = BASE_DIR / "model" / "salary_poly_model.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
departments = sorted(df["Department"].dropna().unique().tolist())
employee_id_min = int(df["Employee_ID"].min())
employee_id_max = int(df["Employee_ID"].max())

app = FastAPI(title="INR Salary Predictor", version="1.0.0")


class PredictionRequest(BaseModel):
    employee_id: int = Field(strict=True, ge=employee_id_min, le=employee_id_max)
    department: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    employee_id: int
    department: str
    monthly_salary_inr: float


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>INR Salary Predictor</title>
    <style>
      body { font: 16px system-ui, sans-serif; margin: 0; padding: 2rem 1rem; background: #f4f7fb; color: #172033; }
      main { max-width: 34rem; margin: 3rem auto; padding: 2rem; background: white; border-radius: 1rem; box-shadow: 0 8px 30px #17203318; }
      h1 { margin-top: 0; }
      label { display: block; margin: 1.25rem 0 .4rem; font-weight: 600; }
      select, input, button { box-sizing: border-box; width: 100%; padding: .8rem; border: 1px solid #cbd5e1; border-radius: .5rem; font: inherit; }
      button { margin-top: 1.5rem; border: 0; background: #1d3557; color: white; font-weight: 700; cursor: pointer; }
      button:disabled { opacity: .65; cursor: wait; }
      #result { margin-top: 1.5rem; font-size: 1.25rem; font-weight: 700; }
      #message { color: #9f1239; }
      #chart { display: block; width: 100%; height: auto; margin-top: 1rem; }
      table { width: 100%; border-collapse: collapse; }
      th, td { padding: .5rem; border-bottom: 1px solid #e2e8f0; text-align: left; }
      .table-wrap { overflow-x: auto; }
    </style>
  </head>
  <body>
    <main>
      <h1>Salary predictor</h1>
      <p>Estimate monthly salary in INR from an employee ID and department.</p>
      <form id="prediction-form">
        <label for="department">Department</label>
        <select id="department" name="department" required></select>
        <label for="employee-id">Employee ID</label>
        <input id="employee-id" name="employee_id" type="number" step="1" required>
        <button id="submit" type="submit">Estimate salary</button>
      </form>
      <p id="result" aria-live="polite"></p>
      <p id="message" role="alert"></p>
      <h2>Salary curve</h2>
      <svg id="chart" viewBox="0 0 720 360" role="img" aria-label="Salary curve for the selected department"></svg>
      <details>
        <summary>View dataset</summary>
        <div class="table-wrap">
          <table>
            <thead id="dataset-head"></thead>
            <tbody id="dataset-body"></tbody>
          </table>
        </div>
      </details>
    </main>
    <script>
      const form = document.getElementById("prediction-form");
      const departmentSelect = document.getElementById("department");
      const employeeIdInput = document.getElementById("employee-id");
      const submitButton = document.getElementById("submit");
      const result = document.getElementById("result");
      const message = document.getElementById("message");
      const chart = document.getElementById("chart");
      const datasetHead = document.getElementById("dataset-head");
      const datasetBody = document.getElementById("dataset-body");
      let departmentData = null;

      function drawChart(prediction = null) {
        if (!departmentData) return;
        chart.replaceChildren();
        const svg = "http://www.w3.org/2000/svg";
        const add = (name, attributes, parent = chart) => {
          const element = document.createElementNS(svg, name);
          for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
          parent.append(element);
          return element;
        };
        const title = add("title", {});
        title.textContent = `Employee ID and monthly salary for ${departmentSelect.value}`;
        const x = (id) => 55 + ((id - departmentData.employee_id_min) /
          (departmentData.employee_id_max - departmentData.employee_id_min)) * 640;
        const maxSalary = Math.max(...departmentData.actual.map((row) => row.Monthly_Salary_INR),
          ...departmentData.curve.map((row) => row.Monthly_Salary_INR), prediction || 0);
        const y = (salary) => 315 - (salary / maxSalary) * 280;
        add("line", { x1: 55, y1: 315, x2: 695, y2: 315, stroke: "#64748b" });
        add("line", { x1: 55, y1: 25, x2: 55, y2: 315, stroke: "#64748b" });
        add("polyline", {
          points: departmentData.curve.map((row) => `${x(row.Employee_ID)},${y(row.Monthly_Salary_INR)}`).join(" "),
          fill: "none", stroke: "#1d3557", "stroke-width": 3
        });
        for (const row of departmentData.actual) {
          add("circle", { cx: x(row.Employee_ID), cy: y(row.Monthly_Salary_INR), r: 3, fill: "#e63946" });
        }
        if (prediction !== null) {
          add("circle", { cx: x(Number(employeeIdInput.value)), cy: y(prediction), r: 7, fill: "#2a9d8f" });
        }
        const xLabel = add("text", { x: 375, y: 350, "text-anchor": "middle", fill: "#172033" });
        xLabel.textContent = "Employee ID";
        const yLabel = add("text", {
          x: 17, y: 170, transform: "rotate(-90 17 170)", "text-anchor": "middle", fill: "#172033"
        });
        yLabel.textContent = "Monthly salary (INR)";
      }

      async function loadDepartmentData() {
        const response = await fetch(`/api/department-data?department=${encodeURIComponent(departmentSelect.value)}`);
        if (!response.ok) throw new Error("Could not load the salary curve.");
        departmentData = await response.json();
        drawChart();
      }

      async function loadDataset() {
        const response = await fetch("/api/dataset");
        if (!response.ok) throw new Error("Could not load the dataset.");
        const rows = await response.json();
        if (!rows.length) return;
        const columns = Object.keys(rows[0]);
        const header = document.createElement("tr");
        for (const column of columns) {
          const cell = document.createElement("th");
          cell.textContent = column;
          header.append(cell);
        }
        datasetHead.append(header);
        for (const row of rows) {
          const tableRow = document.createElement("tr");
          for (const column of columns) {
            const cell = document.createElement("td");
            cell.textContent = row[column];
            tableRow.append(cell);
          }
          datasetBody.append(tableRow);
        }
      }

      async function loadOptions() {
        const response = await fetch("/api/options");
        if (!response.ok) throw new Error("Could not load predictor options.");
        const options = await response.json();
        for (const department of options.departments) {
          const option = document.createElement("option");
          option.value = department;
          option.textContent = department;
          departmentSelect.append(option);
        }
        employeeIdInput.min = options.employee_id_min;
        employeeIdInput.max = options.employee_id_max;
        employeeIdInput.value = options.employee_id_min;
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        result.textContent = "";
        message.textContent = "";
        submitButton.disabled = true;
        try {
          const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              employee_id: Number(employeeIdInput.value),
              department: departmentSelect.value
            })
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
          }
          result.textContent = `Estimated monthly salary: ₹${data.monthly_salary_inr.toLocaleString("en-IN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
          })}`;
          drawChart(data.monthly_salary_inr);
        } catch (error) {
          message.textContent = error.message;
        } finally {
          submitButton.disabled = false;
        }
      });

      departmentSelect.addEventListener("change", () => {
        result.textContent = "";
        loadDepartmentData().catch((error) => {
          message.textContent = error.message;
        });
      });

      loadDataset().catch((error) => {
        message.textContent = error.message;
      });
      loadOptions().then(loadDepartmentData).catch((error) => {
        message.textContent = error.message;
        submitButton.disabled = true;
      });
    </script>
  </body>
</html>"""


@app.get("/api/options")
def get_options() -> dict[str, object]:
    return {
        "departments": departments,
        "employee_id_min": employee_id_min,
        "employee_id_max": employee_id_max,
    }


@app.get("/api/department-data")
def get_department_data(department: str) -> dict[str, object]:
    if department not in departments:
        raise HTTPException(status_code=422, detail="Choose a department from the available options.")

    department_rows = df[df["Department"] == department]
    grid_ids = np.linspace(employee_id_min, employee_id_max, 200)
    grid_data = pd.DataFrame(
        {"Employee_ID": grid_ids, "Department": [department] * len(grid_ids)}
    )
    curve = model.predict(grid_data)
    return {
        "employee_id_min": employee_id_min,
        "employee_id_max": employee_id_max,
        "actual": department_rows[["Employee_ID", "Monthly_Salary_INR"]].to_dict(orient="records"),
        "curve": [
            {"Employee_ID": float(employee_id), "Monthly_Salary_INR": float(salary)}
            for employee_id, salary in zip(grid_ids, curve)
        ],
    }


@app.get("/api/dataset")
def get_dataset() -> list[dict[str, object]]:
    return df.to_dict(orient="records")


@app.post("/api/predict", response_model=PredictionResponse)
def predict_salary(request: PredictionRequest) -> PredictionResponse:
    if request.department not in departments:
        raise HTTPException(status_code=422, detail="Choose a department from the available options.")

    input_data = pd.DataFrame(
        {"Employee_ID": [request.employee_id], "Department": [request.department]}
    )
    predicted_salary = float(model.predict(input_data)[0])
    return PredictionResponse(
        employee_id=request.employee_id,
        department=request.department,
        monthly_salary_inr=predicted_salary,
    )
