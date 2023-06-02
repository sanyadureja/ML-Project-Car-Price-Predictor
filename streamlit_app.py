import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("LinearRegression.pkl", 'rb'))
car = pd.read_csv('Cleaned Car.csv')

def predict_price(company, car_model, year, fuel_type, kms_driven):
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1, 5)))
    return prediction[0]

def main():
    st.title("Car Price Predictor Analysis")

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    companies.insert(0, 'Select Company')
    years.insert(0, 'Select Year')
    car_models.insert(0, 'Select Car Model')
    fuel_types.insert(0, 'Select Fuel Type')

    company = st.selectbox("Select The Company", companies)
    car_model = st.selectbox("Select The Model", car_models)
    year = st.selectbox("Select The Year of Purchase", years)
    fuel_type = st.selectbox("Select The Fuel Type", fuel_types)
    kms_driven = st.number_input("Enter The Number of Kilometres Driven By The Car")

    if st.button("Predict Price Of Car"):
        if company != 'Select Company' and car_model != 'Select Car Model' and year != 'Select Year' and fuel_type != '' and kms_driven != '':
            prediction = predict_price(company, car_model, year, fuel_type, kms_driven)
            st.success(f"Predicted Price: ₹{round(prediction, 2)}")
        else:
            st.error("Please select all the required options.")

if __name__ == '__main__':
    main()
