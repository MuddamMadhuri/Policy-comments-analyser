import pandas as pd
import numpy as np
import joblib
import os
import sys
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Configuration
MODEL_DIR = "policy_analysis/models"
SENTIMENT_RESULT_DIR = "policy_analysis/models_improved/sentiment_model_v2"
INFERENCE_FILE = os.path.join(SENTIMENT_RESULT_DIR, "inference_explainability.csv")
TEST_DATA_FILE = os.path.join(SENTIMENT_RESULT_DIR, "test_data.csv")
TOPIC_MODEL_PATH = os.path.join(MODEL_DIR, "topic_model.joblib")
TOPIC_TFIDF_PATH = os.path.join(MODEL_DIR, "topic_tfidf.joblib")
TOPIC_ENCODER_PATH = os.path.join(MODEL_DIR, "topic_encoder.joblib")

def load_data():
    """Loads inference results and aligns with raw text."""
    if not os.path.exists(INFERENCE_FILE):
        print(f"Error: Inference file not found at {INFERENCE_FILE}")
        sys.exit(1)
    
    print(">>> Loading Data & Models...")
    # Load predictions
    df_pred = pd.read_csv(INFERENCE_FILE)
    
    # Load raw text (assuming 1:1 alignment as established in pipeline)
    if os.path.exists(TEST_DATA_FILE):
        df_text = pd.read_csv(TEST_DATA_FILE)
        # Verify alignment safety
        if len(df_pred) == len(df_text):
            df_pred['comment_text'] = df_text['comment_text']
        else:
            print(f"Warning: Length mismatch (Preds: {len(df_pred)}, Text: {len(df_text)}). Using snippets only.")
            df_pred['comment_text'] = df_pred['text_snippet'] # Fallback
    else:
         df_pred['comment_text'] = df_pred['text_snippet']

    return df_pred

def predict_topics(df, topic_model, vectorizer, encoder):
    """Predicts topics for the comments."""
    print(">>> Classifying Themes...")
    if 'comment_text' not in df.columns:
        return df

    # Transform text
    try:
        X = vectorizer.transform(df['comment_text'].fillna(""))
        topics_idx = topic_model.predict(X)
        topic_labels = encoder.inverse_transform(topics_idx)
        df['topic'] = topic_labels
    except Exception as e:
        print(f"Topic prediction failed: {e}")
        df['topic'] = "Unknown"
        
    return df

def generate_sentiment_overview(df):
    """Generates Section 1: Policy Sentiment Overview."""
    total = len(df)
    counts = df['final_label'].value_counts()
    
    print("\n" + "="*50)
    print("1️⃣  POLICY SENTIMENT OVERVIEW")
    print("="*50)
    
    normalization = 100 / total
    dist = {}
    for label in ["Positive", "Negative", "Neutral", "Mixed"]:
        count = counts.get(label, 0)
        pct = count * normalization
        dist[label] = pct
        print(f"{label:<10}: {pct:>5.1f}%")
        
    print("\n[Interpretation]")
    if dist['Negative'] > 50:
        print("CRITICAL: Public intent alignment is LOOSE. Major revisions strongly required.")
    elif dist['Negative'] > 35:
        print("WARNING: Public intent alignment is MODERATE. Execution concerns are significant.")
    elif dist['Positive'] > 50:
        print("SUCCESS: Strong public mandate detected. Proceed with implementation.")
    else:
        print("OBSERVATION: Divided or uncertain public response. Targeted communication needed.")
        
    return dist

def generate_theme_analysis(df):
    """Generates Section 2: Theme-Based Concern and Support."""
    print("\n" + "="*50)
    print("2️⃣  THEME-BASED CONCERN & SUPPORT ANALYSIS")
    print("="*50)
    
    if 'topic' not in df.columns:
        print("Topic data unavailable.")
        return

    # Group by Topic and Sentiment
    summary = df.groupby(['topic', 'final_label']).size().unstack(fill_value=0)
    
    # Calculate dominant sentiment per topic
    print(f"{'THEME':<30} | {'DOMINANT SENTIMENT':<20} | {'KEY INSIGHT'}")
    print("-" * 80)
    
    for topic in summary.index:
        row = summary.loc[topic]
        total_topic = row.sum()
        if total_topic < 10: continue # Skip tiny topics
        
        dominant = row.idxmax()
        pct = (row[dominant] / total_topic) * 100
        
        insight = ""
        if dominant == "Negative":
            insight = "High Friction Area"
        elif dominant == "Positive":
            insight = "Strong Support Pillar"
        elif dominant == "Mixed":
            insight = "Complex/Nuanced View"
        elif dominant == "Neutral":
            insight = "Information Seeking / Indifferent"
            
        print(f"{topic[:30]:<30} | {dominant} ({pct:.0f}%)      | {insight}")
        
    # Extract examples for top Negative/Mixed topics
    print("\n[Representative Examples]")
    neg_topics = summary['Negative'].sort_values(ascending=False).head(1).index.tolist()
    for t in neg_topics:
        example = df[(df['topic'] == t) & (df['final_label'] == "Negative")]['comment_text'].iloc[0]
        print(f"WARNING ({t}): \"{example[:100]}...\"")

def analyze_mixed_sentiment(df):
    """Generates Section 3: Mixed Sentiment Interpretation."""
    print("\n" + "="*50)
    print("3️⃣  MIXED SENTIMENT INTERPRETATION (WARNING SIGNALS)")
    print("="*50)
    
    mixed_df = df[df['final_label'] == "Mixed"]
    
    if len(mixed_df) == 0:
        print("No Mixed sentiment detected in this batch.")
        return

    print(f"Detected {len(mixed_df)} Mixed comments ({len(mixed_df)/len(df)*100:.1f}% of volume).")
    print("Interpretation: These stakeholders support the GOAL but fear the EXECUTION.")
    
    # Simple N-gram extraction for context
    try:
        vec = CountVectorizer(stop_words='english', ngram_range=(2,3), max_features=5)
        X = vec.fit_transform(mixed_df['comment_text'])
        grams = vec.get_feature_names_out()
        print("\nCommon Reservations / Conditions:")
        for g in grams:
            print(f"- \"...{g}...\"")
    except:
        print("Could not extract snippets.")

def generate_risk_assessment(dist):
    """Generates Section 4: Policy Risk Indicators."""
    print("\n" + "="*50)
    print("4️⃣  POLICY RISK INDICATORS")
    print("="*50)
    
    # Resistance Risk
    res_risk = "LOW"
    if dist['Negative'] > 40: res_risk = "HIGH"
    elif dist['Negative'] > 20: res_risk = "MODERATE"
    
    # Implementation Risk (Driven by Mixed + Negative)
    imp_risk = "LOW"
    if (dist['Mixed'] + dist['Negative']) > 50: imp_risk = "HIGH"
    elif (dist['Mixed'] + dist['Negative']) > 30: imp_risk = "MODERATE"
    
    # Awareness (Inverse of Neutral)
    awareness = "HIGH"
    if dist['Neutral'] > 50: awareness = "LOW"
    elif dist['Neutral'] > 30: awareness = "MODERATE"
    
    print(f"- Resistance Risk:     {res_risk}")
    print(f"- Implementation Risk: {imp_risk}")
    print(f"- Awareness Level:     {awareness}")

    return res_risk, imp_risk

def generate_recommendations(res_risk, imp_risk, df):
    """Generates Section 5: Actionable Recommendations."""
    print("\n" + "="*50)
    print("5️⃣  ACTIONABLE POLICY RECOMMENDATIONS")
    print("="*50)
    
    recs = []
    
    if res_risk == "HIGH":
        recs.append("🔴 PAUSE APPROVAL: Significant public resistance detected.")
        recs.append("- Launch a specialized listening campaign for top negative themes.")
    elif imp_risk == "HIGH":
        recs.append("🟠 REVISE EXECUTION PLAN: Intent is accepted, but details are debated.")
        recs.append("- Clarify enforcement mechanisms and compliance timelines.")
        recs.append("- Release a FAQ addressing specific Mixed-sentiment concerns.")
    else:
        recs.append("🟢 PROCEED WITH MONITORING: General sentiment is stable.")
        
    if df[df['final_label'] == "Neutral"].shape[0] / len(df) > 0.40:
        recs.append("- INCREASE OUTREACH: High neutrality suggests confusion or lack of impact awareness.")
        
    for r in recs:
        print(r)

def main():
    # 1. Load Data
    df = load_data()
    
    # 2. Topic Modeling
    try:
        tm = joblib.load(TOPIC_MODEL_PATH)
        tv = joblib.load(TOPIC_TFIDF_PATH)
        te = joblib.load(TOPIC_ENCODER_PATH)
        df = predict_topics(df, tm, tv, te)
    except Exception as e:
        print(f"Warning: Could not load topic models. Theme analysis will be skipped. {e}")
        
    # 3. Generate Report Sections
    dist = generate_sentiment_overview(df)
    generate_theme_analysis(df)
    analyze_mixed_sentiment(df)
    r_risk, i_risk = generate_risk_assessment(dist)
    generate_recommendations(r_risk, i_risk, df)

if __name__ == "__main__":
    main()
