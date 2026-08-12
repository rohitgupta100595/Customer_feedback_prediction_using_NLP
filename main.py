import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from nltk.corpus import stopwords
import string

st.set_page_config(layout='wide')
# Text Processing function
def text_proc(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word not in stopwords.words('english')]


pipeline = joblib.load('sentiment_pipeline.pkl')

st.title('Customer Sentiment Analyzer')

df = pd.read_csv('cleaned_data.csv')

feedbacks = len(df)
positive_feedback = len(df[df['Sentiment']=='Positive'])
negative_feedback = len(df[df['Sentiment']=='Negative'])
feedback_length = round(df['length'].mean(),2)
positive_df = df[df['Sentiment']=='Positive']
negative_df = df[df['Sentiment']=='Negative']

tab1,tab2 = st.tabs(['Dashboard Section','Feedback Section'])
with tab1:
    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(label='Number of Feedbacks',value=feedbacks)

    with col2:
        st.metric(label='Positive Feedbacks',value=positive_feedback)

    with col3:
        st.metric(label='Negative Feedbacks',value=negative_feedback)

    with col4:
        st.metric(label='Average Length of Message',value=feedback_length)


    hist1 = px.histogram(positive_df,x='length',title='Distribution of Word Length of Positive Feedback')
    hist2 = px.histogram(negative_df,x='length',title='Distribution of Word Length of Negative Feedback')
    countplot1 = px.histogram(negative_df,x = 'Location',title = 'Negative Feedbacks by Location')
    donut1 = px.pie(df,names = 'Sentiment',title = 'Feedback by sentiments',hole = 0.7)

    col5,col6 = st.columns(2)
    with col5:
        st.plotly_chart(hist1)

    with col6:
        st.plotly_chart(hist2)

    col7,col8 = st.columns(2)

    with col7:
        st.plotly_chart(countplot1)

    with col8:
        st.plotly_chart(donut1)


with tab2:
    col1,col2 = st.columns(2)
    with col1:
        Text = st.text_area(label='Feedback Box')
        Btn = st.button(label = 'Predict')

    with col2:

        if Btn:
            if Text.strip() == "":
                st.warning("Please enter a complaint.")
            else:
                prediction = pipeline.predict([Text])[0]

                st.write("Predicted Sentiment: ", prediction)

