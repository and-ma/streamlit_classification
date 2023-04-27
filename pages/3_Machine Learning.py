#Libraries
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from sklearn.ensemble      import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#Page configuration
st.set_page_config(
    page_title="ML",
    layout='wide'
)

#Importing
df = pd.read_csv('train.csv')
image = Image.open('im.jpg')

col_select = st.sidebar.multiselect("Select the attributes to consider for the predictions:",
                                    df.columns[:-1],
                                    default='id')

#Data Preparation
x = df.drop(columns='booking_status')
y = df['booking_status']

for c in df.columns:
    if c in x and c not in col_select:
        x.drop(columns=c, inplace=True)
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mms = MinMaxScaler()
for c in x_train.columns:
    x_train[c] = mms.fit_transform(x_train[[c]].values)
    x_test[c] = mms.fit_transform(x_test[[c]].values)

#Layout
st.write("# Data Science!")
st.image(image, width=500)
st.markdown("## A data app using streamlit")

page_names = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'XGBoost', 'All', 'Make a Prediction']
page = st.radio('Choose a machine learning model to test', page_names)

if page=='Logistic Regression':
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_lr = lr.predict(x_test)
    f1_lr = round(100*mt.f1_score(y_test, y_lr),2)
    st.write("This model has an F1 score = {}%".format(f1_lr)) 

elif page=='Support Vector Machine':
    option = st.selectbox(
    'Which type of Support Vector Machine would you like to test?',
    ('Linear', 'Polynomial', 'RBF', 'Sigmoidal'))
    
    if option=='Linear':
        svm1 = SVC(kernel='linear')
        svm1.fit(x_train, y_train)
        y_svm1 = svm1.predict(x_test)
        f1_svm1 = round(100*mt.f1_score(y_test, y_svm1),2)
        st.write("This model has an F1 score = {}%".format(f1_svm1))

    elif option=='Polynomial':
        svm2 = SVC(kernel='poly')
        svm2.fit(x_train, y_train)
        y_svm2 = svm2.predict(x_test)
        f1_svm2 = round(100*mt.f1_score(y_test, y_svm2),2)
        st.write("This model has an F1 score = {}%".format(f1_svm2))

    elif option=='RBF':
        svm3 = SVC(kernel='rbf')
        svm3.fit(x_train, y_train)
        y_svm3 = svm3.predict(x_test)
        f1_svm3 = round(100*mt.f1_score(y_test, y_svm3),2)
        st.write("This model has an F1 score = {}%".format(f1_svm3))

    else:
        svm4 = SVC(kernel='sigmoid')
        svm4.fit(x_train, y_train)
        y_svm4 = svm4.predict(x_test)
        f1_svm4 = round(100*mt.f1_score(y_test, y_svm4),2)
        st.write("This model has an F1 score = {}%".format(f1_svm4))

elif page=='Random Forest':
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(x_train, y_train)
    y_rf = rf.predict(x_test)
    f1_rf = round(100*mt.f1_score(y_test, y_rf),2)
    st.write("This model has an F1 score = {}%".format(f1_rf))
    
elif page=='XGBoost':
    xgb = xgb.XGBClassifier(n_estimators = 500, max_depth = 10)
    xgb.fit(x_train, y_train)
    y_xgb = xgb.predict(x_test)
    f1_xgb = round(100*mt.f1_score(y_test, y_xgb),2)
    st.write("This model has an F1 score = {}%".format(f1_xgb))
    
elif page=='All':
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_lr = lr.predict(x_test)
    f1_lr = round(100*mt.f1_score(y_test, y_lr),2)
    svm1 = SVC(kernel='linear')
    svm1.fit(x_train, y_train)
    y_svm1 = svm1.predict(x_test)
    f1_svm1 = round(100*mt.f1_score(y_test, y_svm1),2)
    svm2 = SVC(kernel='poly')
    svm2.fit(x_train, y_train)
    y_svm2 = svm2.predict(x_test)
    f1_svm2 = round(100*mt.f1_score(y_test, y_svm2),2)
    svm3 = SVC(kernel='rbf')
    svm3.fit(x_train, y_train)
    y_svm3 = svm3.predict(x_test)
    f1_svm3 = round(100*mt.f1_score(y_test, y_svm3),2)
    svm4 = SVC(kernel='sigmoid')
    svm4.fit(x_train, y_train)
    y_svm4 = svm4.predict(x_test)
    f1_svm4 = round(100*mt.f1_score(y_test, y_svm4),2)
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(x_train, y_train)
    y_rf = rf.predict(x_test)
    f1_rf = round(100*mt.f1_score(y_test, y_rf),2)
    xgb = xgb.XGBClassifier(n_estimators = 500, max_depth = 10)
    xgb.fit(x_train, y_train)
    y_xgb = xgb.predict(x_test)
    f1_xgb = round(100*mt.f1_score(y_test, y_xgb),2)
    
    Y = pd.DataFrame()
    Y['Real Value'] = y_test
    Y['Logistic Prediction'] = y_lr
    Y['Linear Vector Prediction'] = y_svm1
    Y['Polynomial Vector Prediction'] = y_svm2
    Y['RBF Vector Prediction'] = y_svm3
    Y['Sigmoidal Vector Prediction'] = y_svm4
    Y['Random Forest Prediction'] = y_rf
    Y['XGBoost Prediction'] = y_xgb
    
    metrics = pd.DataFrame()
    metrics['Model'] = ['Logistic Regression', 'Linear Vector', 'Polynomial Vector', 'RBF Vector', 'Sigmoidal Vector', 'XGBoost']
    metrics['F1 Score %'] = [f1_lr, f1_svm1, f1_svm2, f1_svm3, f1_svm4, f1_rf, f1_xgb]
    
    option2 = st.selectbox(
    'How would you like to compare the models?',
    ('Table', 'F1 Score'))
    
    if option2=='Table':
        Y
    else:
        metrics
        
else:
    X = pd.DataFrame()
    values = {c: st.number_input('Enter value of {}'.format(c)) for c in x_train.columns}
    X = X.append(values, ignore_index=True)
    X
    
    page_names = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'XGBoost', 'All']
    page = st.radio('Choose a machine learning model to make the prediction', page_names)

    if page=='Logistic Regression':
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_lr = lr.predict(X)
        st.write("The prediction is: {}".format(y_lr)) 

    elif page=='Support Vector Machine':
        option = st.selectbox(
        'Which type of Support Vector Machine would you like to make a prediction?',
        ('Linear', 'Polynomial', 'RBF', 'Sigmoidal'))

        if option=='Linear':
            svm1 = SVC(kernel='linear')
            svm1.fit(x_train, y_train)
            y_svm1 = svm1.predict(X)
            st.write("The prediction is: {}".format(y_svm1))

        elif option=='Polynomial':
            svm2 = SVC(kernel='poly')
            svm2.fit(x_train, y_train)
            y_svm2 = svm2.predict(X)
            st.write("The prediction is: {}".format(y_svm2))

        elif option=='RBF':
            svm3 = SVC(kernel='rbf')
            svm3.fit(x_train, y_train)
            y_svm3 = svm3.predict(X)
            st.write("The prediction is: {}".format(y_svm3))

        else:
            svm4 = SVC(kernel='sigmoid')
            svm4.fit(x_train, y_train)
            y_svm4 = svm4.predict(X)
            f1_svm4 = round(100*mt.f1_score(y_test, y_svm4),2)
            st.write("The prediction is: {}".format(y_svm4))

    elif page=='Random Forest':
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(x_train, y_train)
        y_rf = rf.predict(X)
        st.write("The prediction is: {}".format(y_rf))

    elif page=='XGBoost':
        xgb = xgb.XGBClassifier(n_estimators = 500, max_depth = 10)
        xgb.fit(x_train, y_train)
        y_xgb = xgb.predict(X)
        st.write("The prediction is: {}".format(y_xgb))

    else:
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_lr = lr.predict(X)
        st.write("The Logistic Regression prediction is: {}".format(y_lr))
        svm1 = SVC(kernel='linear')
        svm1.fit(x_train, y_train)
        y_svm1 = svm1.predict(X)
        st.write("The Linear Vector prediction is: {}".format(y_svm1))
        svm2 = SVC(kernel='poly')
        svm2.fit(x_train, y_train)
        y_svm2 = svm2.predict(X)
        st.write("The Polynomial Vector prediction is: {}".format(y_svm2))
        svm3 = SVC(kernel='rbf')
        svm3.fit(x_train, y_train)
        y_svm3 = svm3.predict(X)
        st.write("The RBF Vector prediction is: {}".format(y_svm3))
        svm4 = SVC(kernel='sigmoid')
        svm4.fit(x_train, y_train)
        y_svm4 = svm4.predict(X)
        st.write("The Sigmoidal Vector prediction is: {}".format(y_svm4))
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(x_train, y_train)
        y_rf = rf.predict(X)
        st.write("The Random Forest prediction is: {}".format(y_rf))
        xgb = xgb.XGBClassifier(n_estimators = 500, max_depth = 10)
        xgb.fit(x_train, y_train)
        y_xgb = xgb.predict(X)
        st.write("The XGBoost prediction is: {}".format(y_xgb))
        
        Y = pd.DataFrame()
        Y['Logistic Prediction'] = y_lr
        Y['Linear Vector Prediction'] = y_svm1
        Y['Polynomial Vector Prediction'] = y_svm2
        Y['RBF Vector Prediction'] = y_svm3
        Y['Sigmoidal Vector Prediction'] = y_svm4
        Y['Random Forest Prediction'] = y_rf
        Y['XGBoost Prediction'] = y_xgb
        Y
