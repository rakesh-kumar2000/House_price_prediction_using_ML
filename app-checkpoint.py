import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import time

st.title("House Price Prediction Using Machine Learning")

st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*2djb7kXk2I4SNLBHogl87Q.jpeg")

st.write("We developed a project on House Price Prediction using Machine Learning that leverages data analysis and regression models to estimate property values. This project helps users understand market trends, evaluate factors influencing prices, and make informed investment decisions. It demonstrates skills in Python, data preprocessing, feature engineering, and predictive modeling.")


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns = data.feature_names)
df["MedHouseVal"] = data.target

X = df.iloc[:, :-2]
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

st.sidebar.title("Select House Features")
st.sidebar.image("https://i.pinimg.com/originals/93/c7/44/93c744bcde1780c94bb1d3f03991f8a7.gif")

all_value = []

for i in X.columns:
   value  =  st.sidebar.slider(f'Select {i} value')
   all_value.append(value)

final_data = [all_value]
final_data = scaler.transform(final_data)

with open("price_prediction_for_house.pkl", "rb") as f:
  House_Price_prediction = pickle.load(f)
print("Model Loaded Succesfully!!")

price = House_Price_prediction.predict(final_data)[0]

if price:
   with st.spinner("Wait for price prediction..."):
      time.sleep(5)
      st.success(f"The predicted price of the house is ${price*100000:.2f}")