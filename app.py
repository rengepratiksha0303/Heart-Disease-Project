import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction using KNN")

# Load dataset
@st.cache_data
def load_data():
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv("processed.cleveland.data", names=columns)
    df.replace("?", np.nan, inplace=True)
    df = df.dropna()
    df = df.astype(float)

    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Split data
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sidebar input
st.sidebar.header("Model Settings")
k = st.sidebar.slider("Select K (Neighbors)", 1, 20, 5)

# Train model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"✅ Accuracy: **{accuracy:.2f}**")

# User Input
st.subheader("Predict Heart Disease")

user_input = []
for col in X.columns:
    value = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))
    user_input.append(value)

if st.button("Predict"):
    user_data = scaler.transform([user_input])
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")
