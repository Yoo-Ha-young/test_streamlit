# Contents of ~/my_app/Airline_passenger_satisfaction.py
import streamlit as st

def main_page():
    st.markdown("# Airline passenger satisfaction")
    st.sidebar.markdown("# Airline passenger satisfaction")

def Practice():
    st.markdown("# Practice")
    st.sidebar.markdown("# Page 2 ❄️")
    
def APP():
    st.markdown("# Practice")
    st.sidebar.markdown("# Page 2 ❄️")


page_names_to_funcs = {
    "main_page": Airline_passenger_satisfaction,
    "Practice": Practice,
    "APP": APP,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
