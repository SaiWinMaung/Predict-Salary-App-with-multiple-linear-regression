import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


st.set_page_config(layout="wide")

st.title(":dollar: Salary Prediction App")

# read data file and clean duplicate and null data
data = pd.read_csv("Employee_Salary.csv")

data1 = data.drop_duplicates(keep="first")
data1.dropna(how="any", inplace=True)

data1.columns = ["Age", "Gender", "Degree", "Job_Title", "Experience_Years", "Salary"]

le = LabelEncoder()
data1["gender_encode"] = le.fit_transform(data1["Gender"])
data1["degree_encode"] = le.fit_transform(data1["Degree"])
data1["job_encode"] = le.fit_transform(data1["Job_Title"])

scaler = StandardScaler()
data1["age_scaled"] = scaler.fit_transform(data1[["Age"]])
data1["experience_scaled"] = scaler.fit_transform(data1[["Experience_Years"]])


X = data1[
    ["age_scaled", "gender_encode", "degree_encode", "job_encode", "experience_scaled"]
]
y = data1["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Model = LinearRegression()
Model.fit(X_train, y_train)


with st.sidebar:
    st.header("Please Select Features")
    age = st.slider(
        "Select your age",
        int(data1["Age"].min()),
        int(data1["Age"].max()),
        value=int(30),
    )

    gender = st.selectbox("Select your gender", options=data1["Gender"].unique())

    degree = st.selectbox("Select your degree", options=data1["Degree"].unique())

    job_title = st.selectbox("Select your job title", data1["Job_Title"].unique())

    experience_year = st.slider(
        "Select your experience year",
        int(data1["Experience_Years"].min()),
        int(data1["Experience_Years"].max()),
        value=7,
    )

    age_scaled = scaler.transform([[age]])[0][0]
    experience_year_scaled = scaler.transform([[experience_year]])[0][0]

    job_title_encode = le.transform([job_title])[0]

    gender_encode = 1 if gender == "Male" else 0

    if degree == "Bachelor's":
        degree_encode = 0
    elif degree == "Master's":
        degree_encode = 1
    else:
        degree_encode = 2


input_data = np.array(
    [
        [
            age_scaled,
            gender_encode,
            degree_encode,
            job_title_encode,
            experience_year_scaled,
        ]
    ]
)

predicted_salary = round(Model.predict(input_data)[0], 4)
st.header("Please input features to predict salary")

st.markdown(
    f"<span style='font-size:24px;'>Predicted Salary : $ </span>"
    f"<span style='color:green; font-size:24px; font-weight:bold;'>{predicted_salary}</span>",
    unsafe_allow_html=True
)
