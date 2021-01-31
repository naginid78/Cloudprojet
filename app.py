import streamlit as st
import pandas as pd
import seaborn as sns

steam = pd.read_csv('steam.csv')
ca = pd.read_csv('Car_sales.csv')

st.button("Simple Button")
st.text("Une check box")

status = st.radio("Ton statut",('Active','Inactive'))
if status == 'Active':
    st.text("OK t'es Actif(ve)")
else:
    st.warning("Et un petit warning")

st.text("Petite boite de selection")
