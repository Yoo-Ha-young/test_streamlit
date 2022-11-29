# Contents of ~/my_app/streamlit_app.py
import streamlit as st

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
import pandas as pd
import numpy as np
import streamlit as st
import joblib

st.title("항공기 만족도 예측 Practice")
st.subheader("6가지의 머신러닝 모델을 활용하여 변수들을 바꾸어 예측해보기")

# 첫 번째 행
r1_col1, r1_col2, r1_col3 = st.columns(3)

Inflight_wifi_service = r1_col1.slider("Inflight wifi service", 0, 5)

Departure_Arrival_time_convenient = r1_col2.slider("Departure/Arrival time convenient", 0, 5)

Ease_of_Online_booking = r1_col3.slider("Ease of Online booking", 0,5)


# 두번째 행
r2_col1, r2_col2, r2_col3 = st.columns(3)

Gate_location = r2_col1.slider("Gate location",0,5 )
Food_and_drink = r2_col2.slider("Food and drink", 0,5)
Online_boarding = r2_col3.slider("Online boarding", 0,5)

# 세번째 행
r3_col1, r3_col2, r3_col3 = st.columns(3)
Leg_room_service=r3_col1.slider("Leg room service ",0,5)
Baggage_handling =r3_col2.slider("Baggage handling",0,5)
On_board_service =r3_col3.slider("On-board service",0,5)


# 네 번째 행
r4_col1, r4_col2, r4_col3 = st.columns(3)
Seat_comfort =r4_col1.slider("Seat comfort ",0,5)
Inflight_entertainment =r4_col2.slider("Inflight entertainment",0,5)
Check_in_service =r4_col3.slider("Check-in service",0,5)

# 다섯번째 행
r5_col1, r5_col2 = st.columns(2)
Inflight_service =r5_col1.slider("Inflight service ",0,5)
Cleanliness =r5_col2.slider("Cleanliness",0,5)

# 여섯번째 행
r6_col1, r6_col2 = st.columns(2)
Departure_Delay_in_Minutes=r6_col1.slider("Departure_Delay_in_Minutes ",0,30)
Arrival_Delay_in_Minutes=r6_col2.slider("Arrival_Delay_in_Minutes ",0 ,30)

# 예측 버튼
predict_button = st.button("만족도 예측")

st.write("---")

# 예측 결과
model_list = ['LR_model.pkl', 'KNN_model.pkl', 'DT_model.pkl', 'RandomForestClassifier_model.pkl', 'xgb_model.pkl', 'LightGBM_model.pkl']
result_list = ["로지스틱 회귀 결과", "KNN 결과", "결정트리 결과", "랜덤 포레스트 결과", "XGBoost 결과", "LightGBM 결과"]
for i in range(1, 7):
    if predict_button:
        model = joblib.load(model_list[i-1])
        pred = model.predict(np.array([[Inflight_wifi_service, Departure_Arrival_time_convenient,
        Ease_of_Online_booking, Gate_location, Food_and_drink, Online_boarding, Seat_comfort,
        Inflight_entertainment, On_board_service, Leg_room_service, Baggage_handling, Check_in_service, 
        Inflight_service, Cleanliness,Departure_Delay_in_Minutes,Arrival_Delay_in_Minutes]]))
        
        if pred == 1:
            st.metric(result_list[i-1], "Satisfied")
        else:
            st.metric(result_list[i-1], "Dissatisfied")
    
st.write("")                              
st.markdown('**<center><span style="color: MidnightBlue; font-size:250%">Thank You!</span></center>**', unsafe_allow_html=True)

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

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

# streamlit 앱 제목
st.title("항공사 고객 만족도 Machine Learning")

# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

st.header("데이터 확인")
st.table(airline.head(10))
st.write("원 데이터셋에 약 10만개의 데이터가 있으며, 훈련셋에는 약 26,000개의 데이터가 있다.")

st.write("")
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

logic.logic_reg(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

tree.decision_(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

forest.random_(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

xg.xg_ensemble(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

st.header("LightGBM 학습 결과")

evaluation_list = joblib.load("lightgbm_evaluation.pkl")

st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[0]), unsafe_allow_html=True)
st.markdown('<span style="color: PaleVioletRed; font-size:120%">**정밀도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[1]), unsafe_allow_html=True)
st.markdown('<span style="color: LightBlue; font-size:120%">**재현도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[2]), unsafe_allow_html=True)
st.markdown('<span style="color: PaleTurquoise; font-size:120%">**f1 score:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[3]), unsafe_allow_html=True)
st.markdown('<span style="color: DeepSkyBlue; font-size:120%">**roc_auc_score:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[4]), unsafe_allow_html=True)

image1 = Image.open("lightgbm_confusion_matrix.png")
image2 = Image.open("lightgbm_shap.png")
image3 = Image.open("lightgbm_featureimportance.png")

st.image(image1)

st.write("")
st.subheader("LightGBM Shap Value")
st.image(image2)

st.write("")
st.subheader("LightGBM의 feature importance")
st.image(image3)

st.write("")
st.write("")
st.write("")

knn.neighbors(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

st.markdown('**<center><span style="color: MidnightBlue; font-size:250%">Thank You!</span></center>**', unsafe_allow_html=True) 
    
    
    
page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
# Contents of ~/my_app/main_page.py

