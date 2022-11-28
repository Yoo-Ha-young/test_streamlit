import pandas as pd
import numpy as np
import streamlit as st
import joblib     


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)


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
