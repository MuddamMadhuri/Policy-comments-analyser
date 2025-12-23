from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

def get_tfidf_features(train_texts, val_texts, test_texts, max_features=5000):
    """
    Generates TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_val, X_test, vectorizer

def get_bert_tokenizer(model_name='distilbert-base-uncased'):
    """
    Returns the BERT tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
