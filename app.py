import streamlit as st
import pandas as pd
import joblib


model = joblib.load("/mnt/d/HR project/hr_model.pkl")
preprocessor = joblib.load("/mnt/d/HR project/preprocessor.pkl")

st.title("HR Attrition Prediction App 🚀")

st.write("Enter employee data to predict if they will leave or stay")



Age = st.number_input("Age", 18, 65, 30)

BusinessTravel = st.selectbox(
    "Business Travel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)

Department = st.selectbox(
    "Department",
    ["Sales", "Research & Development", "Human Resources"]
)

EducationField = st.selectbox(
    "Education Field",
    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

JobRole = st.selectbox(
    "Job Role",
    ["Sales Executive", "Research Scientist", "Laboratory Technician",
     "Manufacturing Director", "Healthcare Representative", "Manager"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

DailyRate = st.number_input("Daily Rate", 0)

DistanceFromHome = st.number_input("Distance From Home", 0)

Education = st.number_input("Education Level", 1, 5, 3)

EnvironmentSatisfaction = st.number_input("Environment Satisfaction", 1, 4, 3)

HourlyRate = st.number_input("Hourly Rate", 0)

JobInvolvement = st.number_input("Job Involvement", 1, 4, 3)

JobLevel = st.number_input("Job Level", 1, 5, 2)

JobSatisfaction = st.number_input("Job Satisfaction", 1, 4, 3)

MonthlyIncome = st.number_input("Monthly Income", 0)

MonthlyRate = st.number_input("Monthly Rate", 0)

NumCompaniesWorked = st.number_input("Num Companies Worked", 0)

PercentSalaryHike = st.number_input("Percent Salary Hike", 0)

PerformanceRating = st.number_input("Performance Rating", 1, 4, 3)

RelationshipSatisfaction = st.number_input("Relationship Satisfaction", 1, 4, 3)

StockOptionLevel = st.number_input("Stock Option Level", 0, 3)

TotalWorkingYears = st.number_input("Total Working Years", 0)

TrainingTimesLastYear = st.number_input("Training Times Last Year", 0)

WorkLifeBalance = st.number_input("Work Life Balance", 1, 4, 3)

YearsAtCompany = st.number_input("Years At Company", 0)

YearsInCurrentRole = st.number_input("Years In Current Role", 0)

YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0)

YearsWithCurrManager = st.number_input("Years With Curr Manager", 0)

OverTime = st.selectbox("OverTime", ["Yes", "No"])

# --------- PREDICT ---------

if st.button("Predict Attrition"):

    input_data = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "DailyRate": DailyRate,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "OverTime": OverTime
    }])

    # preprocessing 
    processed = preprocessor.transform(input_data)

    prediction = model.predict(processed)

    if prediction[0] == 1:
        st.error("⚠ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")
