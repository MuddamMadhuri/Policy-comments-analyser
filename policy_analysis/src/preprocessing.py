import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
print("Checking NLTK data...")
try:
    nltk.data.find('corpora/stopwords')
    print(" - stopwords found.")
except LookupError:
    print(" - stopwords not found. Downloading...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
    print(" - wordnet found.")
except LookupError:
    print(" - wordnet not found. Downloading...")
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
    print(" - omw-1.4 found.")
except LookupError:
    print(" - omw-1.4 not found. Downloading...")
    nltk.download('omw-1.4')
print("NLTK data check complete.")

def clean_text(text):
    """
    Cleans text by lowercasing, removing punctuation, stopwords, and lemmatizing.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize (simple split)
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def preprocess_dataframe(df, text_col='comment_text'):
    """
    Applies text cleaning to the dataframe.
    """
    print("Cleaning text...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    return df

def encode_labels(df, columns):
    """
    Encodes categorical labels. Returns the dataframe and the encoders.
    """
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def split_data(df, target_col, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits data into Train, Validation, and Test sets.
    """
    # First split: Train + Val vs Test
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
    
    # Second split: Train vs Val
    # Adjust val_size to be relative to the original dataset size
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, stratify=train_val[target_col], random_state=random_state)
    
    return train, val, test
