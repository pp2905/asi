# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "new_model.sv"
model = pickle.load(open(filename, "rb"))
# otwieramy wcześniej wytrenowany model

gender = {0: "Female", 1: "Male"}
senior_citizen = {0: "No", 1: "Yes"}
partner = {0: "No", 1: "Yes"}
dependents = {0: "No", 1: "Yes"}
phone_service = {0: "No", 1: "Yes"}
multiple_lines = {0: "No", 2: "Yes", 1: "No phone service"}
internet_service = {0: "DSL", 1: "Fiber optic", 2: "No"}
online_security = {0: "No", 2: "Yes", 1: "No internet service"}
online_backup = {0: "No", 2: "Yes", 1: "No internet service"}
device_protection = {0: "No", 2: "Yes", 1: "No internet service"}
tech_support = {0: "No", 2: "Yes", 1: "No internet service"}
streaming_tv = {0: "No", 2: "Yes", 1: "No internet service"}
streaming_movies = {0: "No", 2: "Yes", 1: "No internet service"}
contract = {0: "Month-to-month", 1: "One year", 2: "Two year"}
paperless_billing = {0: "No", 1: "Yes"}
payment_method = {0: "Bank transfer (automatic)", 1: "Credit card (automatic)", 2: "Electronic check", 3: "Mailed check"}


def main():

    st.set_page_config(page_title="App Choroba")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://i1.kwejk.pl/k/obrazki/2017/12/bad1a96d3e90c2f93be2e0c2e67f1878.jpg")

    with overview:
        st.title("App Choroba")

    with left:
        gender_radio = st.radio("Gender", list(gender.keys()), format_func=lambda x: gender[x])
        senior_citizen_radio = st.radio("Senior Citizen", list(senior_citizen.keys()), format_func=lambda x: senior_citizen[x], )
        partner_radio = st.radio("Partner", list(partner.keys()), format_func=lambda x: partner[x], )
        dependents_radio = st.radio("Dependents", list(dependents.keys()), format_func=lambda x: dependents[x], )
        phone_service_radio = st.radio("Phone service", list(phone_service.keys()), format_func=lambda x: phone_service[x], )
        multiple_lines_radio = st.radio("Multiple lines", list(multiple_lines.keys()), format_func=lambda x: multiple_lines[x], )
        internet_service_radio = st.radio("Internet service", list(internet_service.keys()), format_func=lambda x: internet_service[x], )
        online_security_radio = st.radio("Online security", list(online_security.keys()), format_func=lambda x: online_security[x], )
        online_backup_radio = st.radio("Online backup", list(online_backup.keys()), format_func=lambda x: online_backup[x], )
        device_protection_radio = st.radio("Device protection", list(device_protection.keys()), format_func=lambda x: device_protection[x], )
        tech_support_radio = st.radio("Tech support", list(tech_support.keys()), format_func=lambda x: tech_support[x], )
        streaming_tv_radio = st.radio("Streaming TV", list(streaming_tv.keys()), format_func=lambda x: streaming_tv[x], )
        streaming_movies_radio = st.radio("Streaming movies", list(streaming_movies.keys()), format_func=lambda x: streaming_movies[x], )
        contract_radio = st.radio("Contract", list(contract.keys()), format_func=lambda x: contract[x], )
        paperless_billing_radio = st.radio("Paperless billing", list(paperless_billing.keys()), format_func=lambda x: paperless_billing[x], )
        payment_method_radio = st.radio("Payment Method", list(payment_method.keys()), format_func=lambda x: payment_method[x], )

    with right:
        tenure_slider = st.slider("Tenure", value=1, min_value=1, max_value=72)
        monthly_charges_slider = st.slider("Monthly Charges", value=18, min_value=18, max_value=119)
        total_charges_slider = st.slider("Total Charges", value=18, min_value=18, max_value=8685)
        # wzrost_slider = st.slider(
        #     "Wzrost", min_value=150, max_value=215
        # )
        # choroby = st.slider(
        #     "ile chiorób współistniejących", min_value=0, max_value=5
        # )

    data = [
        [
            gender_radio,
            senior_citizen_radio,
            partner_radio,
            dependents_radio,
            tenure_slider,
            phone_service_radio,
            multiple_lines_radio,
            internet_service_radio,
            online_security_radio,
            online_backup_radio,
            device_protection_radio,
            tech_support_radio,
            streaming_tv_radio,
            streaming_movies_radio,
            contract_radio,
            paperless_billing_radio,
            payment_method_radio,
            monthly_charges_slider,
            total_charges_slider,

        ]
    ]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba odejdzie z tego świata")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write(
            "Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100)
        )


if __name__ == "__main__":
    main()
