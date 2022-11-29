# Contents of ~/my_app/Airline_passenger_satisfaction.py
import streamlit as st
st.sidebar.markdown("# K-Means")
st.title("K-Means")

# 필요 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import kmeans
import prepro
import logic
import tree
import forest
import xg
import knn
import joblib
from PIL import Image
import xgboost as xgb
import shap

# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

#st.write("")
st.write("")
st.write("")

airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")
airline2.dropna(inplace=True)

airline.dropna(inplace=True)
    
kmeans.kmeans_clustering(airline, airline2)   
    
X, y, airline_test_X, airline_test = prepro.preprocess(airline, airline2)
    
st.write("")
st.write("")
st.write("")
