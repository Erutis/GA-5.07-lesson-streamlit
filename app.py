import streamlit as st
import plotly.express as px
import pickle
import numpy as np
import time
from predict import predict_flower
#literally just the file predict.py

st.title("Iris Species Predictor")
st.header("Let's predict Iris species")
st.subheader("Cool")

@st.cache() 
def read_data():
    return px.data.iris()


show_df = st.checkbox("Do you want to see the data?")

df_iris = read_data()


col1, col2, col3 = st.columns(3)

with col1:
    sl = st.number_input("Sepal Length (cm)", 0.0, 10.0)

with col2: 
    sw = st.number_input("Sepal Width (cm)", 0.0, 100.0)

with col3: 
    pl = st.number_input("Petal Length (cm)", 0.0, 100.0)
    pw = st.number_input("Petal Width (cm)", 0.0, 100.0)


fig = px.histogram(df_iris, x='sepal_length')
fig

if show_df:
    df_iris



user_input = np.array([[sl, sw, pl, pw]])

with open("saved-iris-model.pkl", "rb") as f:
    time.sleep(1)
    classifier = pickle.load(f)

with st.spinner("predicting"):

    prediction = predict_flower(classifier, user_input)
    prediction

if prediction == 'setosa':
    st.image('https://static.streamlit.io/examples/dog.jpg')

st.balloons()