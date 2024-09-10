# 실행
# streamlit run 5.4.py
import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import pandas as pd

from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('C:\\Users\\노준영\\PycharmProjects\\pythonProject')

img = Image.open('./kt_logo.png')

col1, col2, col3 = st.columns(3)
with col2:
    st.image(img)

# 타이틀
st.title("streamlit 실습")

# 데이터 로딩
loading_progress = st.text('보스턴 집값 데이터 불러오는 중...')
boston = pd.read_csv('./boston_housing.csv')
loading_progress.success('보스턴 집값 데이터 로딩 완료! :house:')

# 데이터프레임 확인
# checkbox 위젯
if st.checkbox('원본 데이터 확인'):
    nrows = st.slider('총 %s개의 데이터 중 몇 개를 보여줄까요?' % boston.shape[0])
    st.subheader('보스턴 집값 원본 데이터')
    # st.table(boston[:nrows])
    # st.write(boston[:nrows])
    st.dataframe(boston[:nrows]) # dataframe은 interactive 기능을 제공

# 변수 시각화
st.header('변수의 단변량 & 다변량 시각화')

# selectbox 위젯
plot_list = ['집값 / 방 개수', '집값 / 방 개수 / 빈공층 비율']
choice = st.selectbox('원하는 변수의 조합을 선택하세요', plot_list)

if choice == plot_list[0]:
    fig = sns.pairplot(boston[['MEDV', 'RM']])
else:
    fig = sns.pairplot(boston[['MEDV', 'RM', 'LSTAT']])

st.pyplot(fig) # figure 출력

# 주택 가격 예측 모델
st.header('주택 가격 금액 예측')

'''
# loading the model
model = joblib.load('./reg_model_boston.pkl')
scaler = joblib.load('./scaler_x.pkl')
'''
st.markdown('*아래의 회귀식으로부터 예측 진행*')


chas_list = ['위치', '위치안함']

CHAS = st.radio('찰스강 경계 위치 여부 입력', chas_list)

if CHAS == '위치':
    CHAS = 1
else:
    CHAS = 0

AGE = st.number_input('오래된 주택 비율 입력 (0~100 사이의 값 입력', min_value=0.00, max_value=100.00, step=1.00)
RM = st.number_input('평균 반의 개수 입력(단위: 개)', min_value=1.00, max_value=100.00, step=1.00)
LSTAT = st.number_input('하위계층 비율 입력 (0~100 사이의 값 입력', min_value=0.00, max_value=100.00, step=1.00)
INDUS = st.number_input('주거지역 토지 비율 입력 (0~100 사이의 값 입력', min_value=0.00, max_value=100.00, step=1.00)
'''
if st.button('Town 주택 가격 예측'):
    new_data = np.array([RM, LSTAT, INDUS, AGE, CHAS])
    new_data = new_data.reshape(1, 5)
    new_data = scaler.transform(new_data)
    Y_pred = model.predict(new_data)
    Y_pred_new = Y_pred + 10000
    round_Y_pred_new = round(Y_pred_new[0,0], 3)

    st.success('이 town의 예측된 주택 가격은 ' + str(round_Y_pred_new) + '달러 입니다. :house:')
'''