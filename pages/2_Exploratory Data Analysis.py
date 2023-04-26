#Libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

#Page configuration
st.set_page_config(
    page_title="EDA",
    layout='wide'
)

#Importing
df = pd.read_csv('train.csv')
image = Image.open('im.jpg')
num_var = df.drop(columns=['id','type_of_meal_plan','required_car_parking_space','room_type_reserved','market_segment_type','repeated_guest','booking_status'])
cat_var = df.loc[:,['type_of_meal_plan','required_car_parking_space','room_type_reserved','market_segment_type','repeated_guest']]

for c in cat_var.columns:
    cat_var[c] = cat_var[c].astype(str)

#Layout
st.write("# Data Science!")
st.image(image, width=500)
st.markdown("## A data app using streamlit")

page_names = ['Univariate', 'Bivariate', 'Multivariate']
page = st.radio('Navigation', page_names)

if page=='Univariate':
    
    b1 = st.button("Target Variable")
    b2 = st.button("Numerical Variables")
    b3 = st.button("Categorical Variables")
    
    if b1:
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.title("Target Variable")
                df_aux = df.loc[:,['booking_status','no_of_adults']].groupby("booking_status").count().reset_index()
                fig = px.pie(df_aux, values='no_of_adults', names='booking_status')
                st.plotly_chart(fig,use_container_width=True)
            
    if b2:

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.title("Numerical Variables")

                for c in num_var.columns:                
                    fig = px.box(df, y=c)
                    st.plotly_chart(fig,use_container_width=True)
                    
    if b3:
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.title("Categorical Variables")
                
                for c in cat_var.columns:
                    df_aux = df.loc[:,['id',c]].groupby(c).count().reset_index()
                    fig = px.bar(df_aux, x=c, y='id')
                    st.plotly_chart(fig, use_container_width=True)
                                    
elif page=='Bivariate':
    b4 = st.button("Numerical Variables")
    b5 = st.button("Categorical Variables")
            
    if b4:

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.title("Numerical Variables")

                for c in num_var.columns:                
                    fig = px.violin(df, x='booking_status', y=c, box=True)
                    st.plotly_chart(fig,use_container_width=True)
                    
    if b5:
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.title("Categorical Variables")
                cat_var['booking_status'] = df['booking_status'].astype(str)
                
                for c in cat_var.columns:
                    if c!='booking_status':
                        fig = px.bar(cat_var, x='booking_status', color=c, barmode='stack')
                        st.plotly_chart(fig, use_container_width=True)

else:
    b6 = st.button("Numerical Variables")
    b7 = st.button("Categorical Variables")
            
    if b6:
        num_var['booking_status'] = df['booking_status']
        
        with st.container():
            st.title("Numerical Variables")
            fig = plt.figure(figsize=(20,10))
            correlation = num_var.corr( method='pearson' )
            sns.heatmap( correlation, annot=True )
            st.pyplot(fig)
                    
    if b7:
        with st.container():
            st.title("Categorical Variables")
            fig = plt.figure(figsize=(20,10))
            corr_matrix = pd.DataFrame(index=cat_var.columns, columns=cat_var.columns)

            for i in range(len(cat_var.columns)):
                for j in range(len(cat_var.columns)):
                    corr_matrix.iloc[i,j] = cramers_v(cat_var.iloc[:,i], cat_var.iloc[:,j])

            corr_matrix = corr_matrix.astype(float)
            sns.heatmap(corr_matrix, annot=True)
            st.pyplot(fig)