import os
import streamlit as st

csv_file = 'clean_risks.csv'

st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Does the file exist? {os.path.exists(csv_file)}")

df = pd.read_csv(csv_file)
