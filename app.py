#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


#from tkinter import Toplevel
#from winreg import HKEY_LOCAL_MACHINE
import streamlit as st
import pandas as pd
import pickle
import joblib
# import plotly.express as px
# import plotly.graph_objects as go


st.title("TEST")
    
@st.cache(allow_output_mutation=True)
def load_data():
    data= pd.read_csv("Data/application_test.csv")  #Currently on my local machine
    data.set_index("SK_ID_CURR", inplace=True)
    return data
df_test = load_data()

@st.cache
def load_model():
    loaded_model = pickle.load(open('Data/finalized_model.sav', 'rb'))
    return loaded_model
model=load_model()

@st.cache  
def scaling():
    loaded_scaling = joblib.load(open('Data/scaler.save', 'rb'))
    return loaded_scaling
scaler = scaling()

@st.cache
def preprocessing(df):
    df.dropna(axis=0, inplace=True)
    df = pd.DataFrame(scaler.transform(df.select_dtypes(include="number")))
    df = pd.get_dummies(df)
    return df

df_test = preprocessing(df_test)

def prediction(df):
    client = df.sample()
    pred = model.predict(client)
    proba = model.predict_proba(client)[0][pred[0]]
    return client.index[0], pred[0], round(proba*100,2)

output = prediction(df_test)


if st.button('Predict'):
    
    if output[1] == 1:
        st.write("The client n°:", output[0], "is in group", output[1])
        st.write("Credit allowed")
        st.write("probabilité dans le groupe:", output[2], "%")
    else:
        st.write("The client n°:", output[0], "is in group", output[1])
        st.write("Credit refused")
        st.write("probabilité dans le groupe:", output[2], "%")