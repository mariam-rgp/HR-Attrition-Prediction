# 👩‍💼 HR Employee Attrition Prediction App

A Machine Learning powered web application built with Streamlit to predict whether an employee is likely to leave the company or stay based on HR-related features.

---

## 🚀 Project Overview

Employee attrition is a major challenge for companies.  
This project helps HR departments identify employees at risk of leaving using a trained Machine Learning model.

The solution includes:
- Data preprocessing pipeline
- Machine Learning model (Logistic Regression)
- Interactive Streamlit web application

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Class balancing: Applied to handle imbalanced data
- Preprocessing pipeline:
  - OneHotEncoding for categorical features
  - MinMax Scaling for numerical features
- Model trained and saved using joblib / pickle

---

## 📊 Input Features

### 🟡 Categorical Features:
- BusinessTravel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- OverTime

### 🔵 Numerical Features:
- Age
- DailyRate
- DistanceFromHome
- Education
- EnvironmentSatisfaction
- HourlyRate
- JobInvolvement
- JobLevel
- JobSatisfaction
- MonthlyIncome
- MonthlyRate
- NumCompaniesWorked
- PercentSalaryHike
- PerformanceRating
- RelationshipSatisfaction
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Matplotlib & Seaborn
- Conda Environment (for dependency management)

---

## 📁 Project Structure
HR project/
│── app.py
│── hr_model.pkl
│── preprocessor.pkl
│── requirements.txt
│── README.md
│── HR_Attrition_Prediction.ipynb

📊 Model Output

The model predicts:

✅ Employee is likely to stay
⚠️ Employee is likely to leave

⚙️ Environment
Developed using Conda environment
Tested on WSL (Windows Subsystem for Linux)
Python 3.10
