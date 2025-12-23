import pandas as pd
import numpy as np
import joblib

def get_top_keywords(vectorizer, model, n=10):
    """
    Extracts top keywords for each class from a linear model.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_
    
    top_keywords = {}
    if coefs.ndim == 1: # Binary case
        # Negative class (0)
        top_negative = [feature_names[i] for i in coefs.argsort()[:n]]
        # Positive class (1)
        top_positive = [feature_names[i] for i in coefs.argsort()[-n:][::-1]]
        top_keywords[0] = top_negative
        top_keywords[1] = top_positive
    else:
        for i, class_coefs in enumerate(coefs):
            top = [feature_names[j] for j in class_coefs.argsort()[-n:][::-1]]
            top_keywords[i] = top
            
    return top_keywords

def analyze_sentiment_by_group(df, group_col, sentiment_col='sentiment'):
    """
    Analyzes sentiment distribution by a grouping column.
    """
    return df.groupby([group_col, sentiment_col]).size().unstack(fill_value=0)

def analyze_stance_by_topic(df, topic_col='topic', stance_col='stance'):
    """
    Analyzes stance distribution by topic.
    """
    return df.groupby([topic_col, stance_col]).size().unstack(fill_value=0)
