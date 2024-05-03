import streamlit as st
import joblib
import pandas as pd


model = joblib.load("E:\\churn\\internet_service_churn.pkl")
df = pd.read_csv("E:\\churn\\internet_service_churn.csv")
df.drop(["churn","id"], axis=1, inplace=True)
df.fillna(0, inplace=True)
input_fields = df.columns
st.title('Churn Rate Prediction')

def get_user_input():
    user_input = {}
    for field in input_fields:
        user_input[field] = st.number_input(f"Enter {field}", step=1)
    return user_input

def predict_churn(user_input):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df[input_fields])
    return prediction[0]

def main():
    user_input = get_user_input()
    if st.button('Predict'):
        prediction = predict_churn(user_input)
        st.write(f"Predicted Churn: {prediction}")

if __name__ == '__main__':
    main()