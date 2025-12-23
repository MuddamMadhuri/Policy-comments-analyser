import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.insights import get_top_keywords, analyze_sentiment_by_group, analyze_stance_by_topic

DATA_PATH = "policy_analysis/data/india_policy_comments_large_50k.csv"
MODEL_DIR = "policy_analysis/models"
OUTPUT_DIR = "policy_analysis/insights"

def run_insights():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading data and models...")
    df = pd.read_csv(DATA_PATH)
    
    # Load Topic Model & Vectorizer to extract keywords
    try:
        topic_model = joblib.load(f"{MODEL_DIR}/topic_model.joblib")
        topic_vectorizer = joblib.load(f"{MODEL_DIR}/topic_tfidf.joblib")
        topic_encoder = joblib.load(f"{MODEL_DIR}/topic_encoder.joblib")
        
        print("\n--- Top Keywords per Topic ---")
        keywords = get_top_keywords(topic_vectorizer, topic_model)
        for class_idx, words in keywords.items():
            topic_name = topic_encoder.inverse_transform([class_idx])[0]
            print(f"{topic_name}: {', '.join(words)}")
            
    except FileNotFoundError:
        print("Topic model files not found. Skipping keyword extraction.")

    # Data-driven insights
    print("\n--- Sentiment by Project ---")
    sent_by_project = analyze_sentiment_by_group(df, 'project_name')
    print(sent_by_project)
    sent_by_project.to_csv(f"{OUTPUT_DIR}/sentiment_by_project.csv")
    
    print("\n--- Stance by Topic ---")
    stance_by_topic = analyze_stance_by_topic(df)
    print(stance_by_topic)
    stance_by_topic.to_csv(f"{OUTPUT_DIR}/stance_by_topic.csv")
    
    print(f"\nInsights saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_insights()
