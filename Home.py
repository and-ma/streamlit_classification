#Libraries
import streamlit as st
import pandas as pd
from PIL import Image

#Page configuration
st.set_page_config(
    page_title="Home Page",
    page_icon="house",
    layout='wide'
)

#Importing
image = Image.open('im.jpg')

#Layout
st.write("# Data Science!")
st.image(image, width=500)
st.markdown("## A data app using streamlit")
