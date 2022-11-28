import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Personal Data 레이블 인코딩 (범주형 데이터를 연속형 변수로 바꿈)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# XGBoost 에 필요한 라이브러리 임포트



# 훈련셋
airline_train = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv", index_col=0)
airline_train.drop(['id'], axis=1, inplace=True)

# 시험셋
airline_test = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv", index_col=0)
airline_test.drop(['id'], axis=1, inplace=True)

# 결측치 제거
airline_train.dropna(inplace=True)
airline_test.dropna(inplace=True)

# 레이블 인코딩
# 범주형 columns 리스트 : temp_list
temp_list = ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]

for i in temp_list:
  # 훈련셋
  encoder.fit(airline_train[i])
  airline_train['{}_score'.format(i)] = encoder.transform(airline_train[i])
  # 시험셋
  encoder.fit(airline_test[i])
  airline_test['{}_score'.format(i)] = encoder.transform(airline_test[i])

# 고객 평가 지표 데이터 프레임
airline_score = airline_train[['Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Cleanliness',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes',
       'satisfaction_score']]


# y, X 정의
X_train = airline_train[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
'Arrival Delay in Minutes']]   # 여기에 특성 추가
X_test = airline_test[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
'Arrival Delay in Minutes']]
y_train = airline_train['satisfaction_score']
y_test = airline_test['satisfaction_score']

X_train.astype(int)
X_test.astype(int)
y_train.astype(int)
y_test.astype(int)

# 평가지표 함수 정의
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluation(airline_test, pred):
    acc = accuracy_score(airline_test, pred)
    pre = precision_score(airline_test, pred)
    rec = recall_score(airline_test, pred)
    f1 = f1_score(airline_test, pred)
    roc = roc_auc_score(airline_test, pred)
    cf_matrix = confusion_matrix(airline_test, pred)
    print("정확도: {0:.4f}".format(acc))
    print("정밀도: {0:.4f}".format(pre))
    print("재현율: {0:.4f}".format(rec))
    print("f1 score: {0:.4f}".format(f1))
    print("roc_auc_score: {0:.4f}".format(roc))
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
       
# 그리드서치 라이브러리 임포트
from sklearn.model_selection import GridSearchCV

def xgb_tuning(train_set, test_set, parameters):
    model = XGBClassifier()
    grid = GridSearchCV(model, parameters, scoring="roc_auc", cv=5, n_jobs=-1, refit = True) # cv=K-fold
    grid.fit(train_set, test_set)
    pred= grid.predict(X_test)
    return grid.best_params_, grid.best_score_, pred

xgb_model = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3, subsample=1)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)

joblib.dump(xgb_model, './xgb_model.pkl')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd


st.title("항공사 고객만족도 XGboost")
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
Inflight_service = r5_col1.slider("Inflight service ",0,5)
Cleanliness = r5_col2.slider("Cleanliness",0,5)
Departure_Delay_in_Minutes = r5_col2.slider("Departure Delay in Minutes",0,5)
Arrival_Delay_in_Minutes = r5_col2.slider("Arrival Delay in Minutes",0,5)


# 예측 버튼
predict_button = st.button("예측")
st.write("---")

# 예측 결과
if predict_button:
    model_from_joblib = joblib.load('xgb_model.pkl')
    pred = model_from_joblib.predict(np.array([['Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Cleanliness',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']]))
    st.metric("예상 만족 여부", pred[0])

