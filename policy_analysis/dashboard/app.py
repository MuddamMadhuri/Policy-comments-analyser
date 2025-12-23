import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Policy Analysis Dashboard", layout="wide")

st.title("Policy Draft Comment Analysis Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Inference", "Insights"])

API_URL = "http://localhost:8000/predict"
DATA_PATH = "policy_analysis/data/india_policy_comments_large_50k.csv"
EDA_OUTPUT_DIR = "policy_analysis/eda_output"
INSIGHTS_DIR = "policy_analysis/insights"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

df = load_data()

if page == "Overview":
    st.header("Project Overview")
    st.write("This dashboard analyzes public comments on policy drafts to extract sentiment, stance, and topics.")
    if df is not None:
        st.write(f"**Total Comments:** {len(df)}")
        st.write("**Dataset Preview:**")
        st.dataframe(df.head())

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        if os.path.exists(f"{EDA_OUTPUT_DIR}/dist_sentiment.png"):
            st.image(f"{EDA_OUTPUT_DIR}/dist_sentiment.png")
            
    with col2:
        st.subheader("Stance Distribution")
        if os.path.exists(f"{EDA_OUTPUT_DIR}/dist_stance.png"):
            st.image(f"{EDA_OUTPUT_DIR}/dist_stance.png")
            
    st.subheader("Topic Distribution")
    if os.path.exists(f"{EDA_OUTPUT_DIR}/dist_topic.png"):
        st.image(f"{EDA_OUTPUT_DIR}/dist_topic.png")
        
    st.subheader("Word Cloud")
    if os.path.exists(f"{EDA_OUTPUT_DIR}/wordcloud.png"):
        st.image(f"{EDA_OUTPUT_DIR}/wordcloud.png")

elif page == "Model Inference":
    st.header("Real-time Prediction")
    st.write("Enter a comment to analyze its sentiment, stance, and topic.")
    
    user_input = st.text_area("Comment Text", "The new policy on renewable energy is a step in the right direction.")
    
    if st.button("Analyze"):
        try:
            # For demo purposes, we can import the prediction logic directly if API is not running
            # But ideally we call the API.
            # Here I'll try to call the API, if it fails, I'll show a message.
            try:
                response = requests.post(API_URL, json={"text": user_input})
                if response.status_code == 200:
                    result = response.json()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sentiment", result['sentiment'])
                    col2.metric("Stance", result['stance'])
                    col3.metric("Topic", result['topic'])
                else:
                    st.error("API Error")
            except requests.exceptions.ConnectionError:
                st.warning("API is not running. Please start the FastAPI server.")
                
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Insights":
    st.header("Key Insights")
    
    if os.path.exists(f"{INSIGHTS_DIR}/sentiment_by_project.csv"):
        st.subheader("Sentiment by Project")
        sent_df = pd.read_csv(f"{INSIGHTS_DIR}/sentiment_by_project.csv")
        st.dataframe(sent_df)
        
    if os.path.exists(f"{INSIGHTS_DIR}/stance_by_topic.csv"):
        st.subheader("Stance by Topic")
        stance_df = pd.read_csv(f"{INSIGHTS_DIR}/stance_by_topic.csv")
        st.dataframe(stance_df)
