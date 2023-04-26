#Libraries
import streamlit as st
import pandas as pd
from PIL import Image

#Page configuration
st.set_page_config(
    page_title="Data",
    layout='wide'
)

#Importing
df = pd.read_csv('train.csv')
image = Image.open('im.jpg')

#Layout
st.write("# Data Science!")
st.image(image, width=500)
st.markdown("## A data app using streamlit")
st.markdown("### Exploring the dataset")

#Checkbox
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
page_names = ['See Data', 'Data Dimensions', 'Data Types', 'Check Null Values', 'Descriptive Statistics']
page = st.radio('Navigation', page_names)

if page=='See Data':
    df
elif page=='Data Dimensions':
    st.markdown('The dataset contains {} rows and {} columns'.format(df.shape[0], df.shape[1]))
elif page=='Data Types':
    df.dtypes
elif page=='Check Null Values':
    st.write(df.isna().sum())
else:
    st.write(df.describe().T)