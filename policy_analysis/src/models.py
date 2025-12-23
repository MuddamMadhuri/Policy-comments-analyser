from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification

def get_baseline_model(model_type='lr', class_weight=None):
    """
    Returns a baseline model (Logistic Regression or SVM).
    """
    if model_type == 'lr':
        return LogisticRegression(max_iter=1000, class_weight=class_weight)
    elif model_type == 'svm':
        return LinearSVC(max_iter=1000, class_weight=class_weight)
    else:
        raise ValueError("Invalid model_type. Choose 'lr' or 'svm'.")

def get_bert_model(model_name='distilbert-base-uncased', num_labels=3):
    """
    Returns a pre-trained BERT model for sequence classification.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
