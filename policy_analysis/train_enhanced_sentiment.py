import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import Dataset

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# File Paths (UPDATE THESE)
# Assuming run from root of workspace or adjusted manually
TRAIN_CSV_PATH = r"c:/Users/madhu/nlp_proj/policy_analysis/data/train.csv"
VAL_CSV_PATH = r"c:/Users/madhu/nlp_proj/policy_analysis/data/val.csv"
RAW_CSV_PATH = r"c:/Users/madhu/nlp_proj/policy_analysis/data/india_policy_comments_large_50k.csv"
OUTPUT_DIR = r"c:/Users/madhu/nlp_proj/policy_analysis/models_improved/sentiment_model_improved_bert"

# Strategy selection
# Options: "oversample" (duplicate minority samples) or "weighted_loss" (class weights in loss)
STRATEGY = "weighted_loss" 

# Model & Training Config
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

# Label Definitions
# Ensure these match your data. If labels are strings in CSV, we will encode them.
# If they are already integers, we assume 0-3 range.
EXPECTED_LABELS = ["Mixed", "Negative", "Neutral", "Positive"]

# ==========================================
# HELPER CLASSES & FUNCTIONS
# ==========================================

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert weights to tensor and move to model device
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Fixed signature
        # Ensure weights are on the same device as the model
        if self.class_weights.device != model.device:
            self.class_weights = self.class_weights.to(model.device)
            
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class TemperatureScaler(nn.Module):
    """
    A specific Temperature Scaling module.
    Scales logits by a single scalar T > 0.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        # Allow passing full input dict or just logits
        # If input is tensor, assume it's logits
        if isinstance(input, torch.Tensor):
            return input / self.temperature
        
        # If input is dict (like from tokenizer), pass through model first
        # But usually we wrap the model, so we might receive inputs typical for model.
        # To avoid confusion, let's keep it simple: 
        if isinstance(input, dict) or hasattr(input, 'items'):
            # If input is dict, pass through model
            outputs = self.model(**input)
            return self.temperature_scale(outputs.logits)
        
        # Fallback for other types if any
        raise ValueError("TemperatureScaler expects logits (Tensor) or model inputs (dict/BatchEncoding)")

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # Validation optimization
    def set_temperature(self, valid_loader, device="cpu"):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        """
        self.to(device)
        self.model.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        
        # First, collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        
        from tqdm import tqdm
        print("Collecting logits for calibration...")
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Calibration"):
                # Move batch to CPU (or specified device, typically CPU to save GPU RAM)
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if 'labels' in batch:
                    labels = batch['labels']
                elif 'label' in batch:
                    labels = batch['label']
                else:
                    raise KeyError("Batch does not contain 'labels' or 'label'")
                
                outputs = self.model(**batch)
                logits = outputs.logits
                
                logits_list.append(logits)
                labels_list.append(labels)
                
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL before scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        print(f"Before temperature - NLL: {before_temperature_nll:.3f}")

        # Next: Optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL after scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        print(f"After temperature - NLL: {after_temperature_nll:.3f}")

        return self


class HybridSentimentPredictor:
    """
    Hybrid inference strategy to improve recall for 'Mixed' sentiment.
    Treats 'Mixed' as a composite label where both Positive and Negative
    signals are strong, overriding the standard argmax.
    """
    def __init__(self, model, tokenizer=None, scaler=None, pos_threshold=0.45, neg_threshold=0.35, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.device = device
        
        # Label Mapping (Hardcoded for this specific pipeline's consistency)
        self.MIXED_IDX = 0
        self.NEG_IDX = 1
        self.NEU_IDX = 2
        self.POS_IDX = 3

    def predict_proba(self, texts_or_loader):
        """
        Get calibrated probabilities.
        Accepts either a list of texts or a DataLoader.
        """
        self.model.eval()
        self.model.to(self.device)
        if self.scaler:
            self.scaler.to(self.device)
            
        all_probs = []
        
        with torch.no_grad():
            if isinstance(texts_or_loader, torch.utils.data.DataLoader):
                for batch in texts_or_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if k in self.model.forward.__code__.co_varnames}
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    if self.scaler:
                        logits = self.scaler.temperature_scale(logits)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    all_probs.append(probs.cpu())
            else:
                # Assume list of strings
                # Chunking would be better for prod, but kept simple here
                inputs = self.tokenizer(texts_or_loader, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                if self.scaler:
                    logits = self.scaler.temperature_scale(logits)
                params = torch.nn.functional.softmax(logits, dim=-1)
                all_probs.append(params.cpu())
                
        return torch.cat(all_probs).numpy()

    def apply_hybrid_rule(self, prob_array):
        """
        Apply the hybrid decision rule on a numpy array of probabilities (N, 4).
        """
        predictions = []
        for p in prob_array:
            # Rule: If Prob(Positive) > T_pos AND Prob(Negative) > T_neg -> Mixed
            if p[self.POS_IDX] >= self.pos_threshold and p[self.NEG_IDX] >= self.neg_threshold:
                predictions.append(self.MIXED_IDX)
            else:
                predictions.append(np.argmax(p))
        return np.array(predictions)


def load_and_prep_data(train_path, val_path):
    print(f"Checking data at:\n Train: {train_path}\n Val: {val_path}")
    
    # Check if split files exist
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Train/Val files not found. Checking raw data at {RAW_CSV_PATH}...")
        if os.path.exists(RAW_CSV_PATH):
            print("Loading raw data and creating splits...")
            df_raw = pd.read_csv(RAW_CSV_PATH)
            
            # Rename columns to match expected format
            # Adjust these column names if your raw csv differs
            if 'comment_text' in df_raw.columns:
                 df_raw = df_raw.rename(columns={'comment_text': 'text', 'sentiment': 'label'})
            
            # Select only needed columns
            df_raw = df_raw[['text', 'label']]
            
            # Split
            df_train, df_val = train_test_split(df_raw, test_size=0.2, stratify=df_raw['label'], random_state=42)
            
            # Save for future use
            print(f"Saving splits to {train_path} and {val_path}")
            df_train.to_csv(train_path, index=False)
            df_val.to_csv(val_path, index=False)
        else:
             raise FileNotFoundError(f"Neither {train_path} nor {RAW_CSV_PATH} found.")
    else:
        print("Loading existing split files...")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
    
    # Ensure columns exist
    if 'text' not in df_train.columns or 'label' not in df_train.columns:
        raise ValueError("CSV files must contain 'text' and 'label' columns.")
        
    # Handle Label Encoding if labels are strings
    if df_train['label'].dtype == 'object':
        le = LabelEncoder()
        le.fit(EXPECTED_LABELS) # Fit on expected to ensure consistency
        df_train['label'] = le.transform(df_train['label'])
        df_val['label'] = le.transform(df_val['label'])
        
        # In case val contains labels not in expected (should not happen if consistent), handle it or let it error
        # Assuming EXPECTED_LABELS covers all
        print("Labels Encoded:", dict(zip(le.classes_, le.transform(le.classes_))))
        
    return df_train, df_val

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    df_train, df_val = load_and_prep_data(TRAIN_CSV_PATH, VAL_CSV_PATH)
    
    print("\nInitial Training Class Distribution:")
    print(df_train['label'].value_counts().sort_index())
    
    classes = np.arange(len(EXPECTED_LABELS))
    
    # 2. Compute Class Weights
    # We calculate manually to ensure coverage of all 4 classes even if some are missing in train split
    class_counts = df_train['label'].value_counts().sort_index()
    counts = np.array([class_counts.get(i, 0) for i in classes])
    
    # Standard formula: n_samples / (n_classes * count)
    # Add epsilon to count to avoid div by zero if a class is completely missing
    n_samples = len(df_train)
    n_classes = len(classes)
    
    weights = n_samples / (n_classes * (counts + 1e-6)) # Avoid inf
    
    # Normalize? Not strictly necessary for CrossEntropyLoss but good for stability 
    # weights = weights / weights.sum() * n_classes 
    
    class_weights = weights
    print("\nComputed Class Weights:", dict(zip(EXPECTED_LABELS, class_weights)))
    
    # 3. Strategy Implementation
    if STRATEGY == "oversample":
        print("\n[Strategy] Applying Oversampling to 'Mixed' class (assuming it is the minority)...")
        # Identify counts
        counts = df_train['label'].value_counts()
        max_count = counts.max()
        
        dfs = []
        for label_val in classes:
            df_subset = df_train[df_train['label'] == label_val]
            if len(df_subset) == 0:
                 continue
            if len(df_subset) < max_count:
                # Upsample
                print(f"Upsampling class {label_val} from {len(df_subset)} to {max_count}")
                df_upsampled = df_subset.sample(max_count, replace=True, random_state=42)
                dfs.append(df_upsampled)
            else:
                dfs.append(df_subset)
        
        df_train = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print("New Training Class Distribution (Oversampled):")
        print(df_train['label'].value_counts().sort_index())
        
        # In oversample strategy, we usually don't verify weighted loss as strictly, but we can combined them.
        # However, the requirement implies "Selectable", so we use standard trainer here.
        trainer_class = Trainer
        trainer_kwargs = {}
        
    elif STRATEGY == "weighted_loss":
        print("\n[Strategy] Using Weighted Loss in CrossEntropy.")
        trainer_class = WeightedLossTrainer
        trainer_kwargs = {"class_weights": class_weights}
    else:
        raise ValueError(f"Unknown strategy: {STRATEGY}")
        
    # 4. Tokenization & Dataset Creation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
    val_dataset = Dataset.from_pandas(df_val[['text', 'label']])
    
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Remove strings and other non-tensor columns to avoid errors
    train_tokenized = train_tokenized.remove_columns(["text"])
    val_tokenized = val_tokenized.remove_columns(["text"])
    if "__index_level_0__" in train_tokenized.column_names:
        train_tokenized = train_tokenized.remove_columns(["__index_level_0__"])
    if "__index_level_0__" in val_tokenized.column_names:
        val_tokenized = val_tokenized.remove_columns(["__index_level_0__"])

    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")

    # 5. Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(classes)
    )
    model.to(device)
    
    # Training Arguments
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        save_total_limit=2,
        # BATCHING EXPLANATION (req 4):
        # We rely on global shuffling (group_by_length=False by default) provided by Trainer.
        # For 'oversample' strategy, the dataset is balanced, so batches are naturally balanced.
        # For 'weighted_loss', we keep the natural distribution but penalize errors more heavily via Class Weights.
        # Strict stratified sampling per-batch is complex in distributed/streaming setups, so we use weighted loss.
        dataloader_num_workers=0, # Crucial for Windows to avoid hangs
    )
    
    # Function to compute metrics during training
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        return {
            'accuracy': (predictions == labels).mean(),
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Initialize Trainer
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        **trainer_kwargs
    )
    
    # Check if we should skip training (if model already exists)
    # We check for config and weights
    has_model = (
        os.path.exists(os.path.join(OUTPUT_DIR, "config.json")) and 
        (os.path.exists(os.path.join(OUTPUT_DIR, "pytorch_model.bin")) or 
         os.path.exists(os.path.join(OUTPUT_DIR, "model.safetensors")))
    )
    
    if has_model:
        print(f"Model found in {OUTPUT_DIR}. Skipping training as per 'Do not retrain' request.")
        # Load the fine-tuned weights
        model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR, num_labels=len(classes))
        model.to(device)
        # Update trainer's model just in case, though we mainly use it for dataloader
        trainer.model = model
    else:
        print("\nStarting Training...")
        trainer.train()
        
        # Save base model
        print(f"\nSaving best model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
    
    # 6. Calibration (Temperature Scaling)
    print("\nCalibrating Probabilities (Temperature Scaling)...")
    
    # We need a DataLoader for the validation set
    # Using the Trainer's processing for consistency
    # (Note: trainer.get_eval_dataloader() handles collation/padding)
    val_loader = trainer.get_eval_dataloader(val_tokenized)
    
    # Initialize Scaler
    scaler = TemperatureScaler(model)
    
    # Move everything to CPU for calibration to avoid OOM
    print("Moving model to CPU for calibration safety...")
    scaler.set_temperature(val_loader, device="cpu")
    
    # Save temperature
    with open(os.path.join(OUTPUT_DIR, "temperature_scaling.txt"), "w") as f:
        f.write(str(scaler.temperature.item()))

    
    # 7. Evaluation
    print("\nEvaluating Calibrated Model...")
    
    # Prediction loop for calibration wrapper (if dataset is large, might want to batch this explicitly, 
    # but the wrapper handles batching inside predict_proba via tokenizer, wait, my wrapper definition above 
    # tokenizes the WHOLE list at once. This is risky for large datasets.
    # Prediction with Temperature Scaling
    # We can reuse the batches logic or just iterate the loader we already have
    # Since we are on CPU/Memory constrained, let's just use the loader
    
    from tqdm import tqdm
    print("Generating predictions with calibrated probabilities...")
    all_probs = []
    
    model.eval()
    model.to("cpu") # Ensure on cpu
    scaler.to("cpu")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
             # Move to CPU
             batch = {k: v.to("cpu") for k, v in batch.items()}
             outputs = model(**batch)
             logits = outputs.logits
             
             # Scale
             scaled_logits = scaler.temperature_scale(logits)
             probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
             all_probs.append(probs)
             
    y_prob = torch.cat(all_probs).cpu().numpy()

    y_pred = np.argmax(y_prob, axis=1)
    
    # Print Classification Report
    print("\nClassification Report (Calibrated):")
    # Map back to string labels if possible
    # We used label encoder earlier if string. If we didn't save the LE, we might miss the mapping.
    # But EXPECTED_LABELS is predefined.
    
    val_labels = df_val['label'].values
    val_texts = df_val['text'].values
    
    report = classification_report(val_labels, y_pred, target_names=EXPECTED_LABELS, zero_division=0)
    print(report)
    
    # Save Report
    report_dict = classification_report(val_labels, y_pred, target_names=EXPECTED_LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(val_labels, y_pred)
    pd.DataFrame(cm, index=EXPECTED_LABELS, columns=EXPECTED_LABELS).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))
    print("\nConfusion Matrix Saved.")
    
    # Plot Confusion Matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EXPECTED_LABELS, yticklabels=EXPECTED_LABELS)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
        plt.close()
        print("Confusion Matrix Plot Saved to confusion_matrix.png")
    except ImportError:
        print("Warning: matplotlib or seaborn not found. Skipping plot.")
    except Exception as e:
        print(f"Warning: Could not plot confusion matrix: {e}")
    
    # 8. Top 10 Analysis for 'Mixed'
    try:
        mixed_idx = EXPECTED_LABELS.index('Mixed')
    except ValueError:
        mixed_idx = 0 
        
    print(f"\nTop 10 'Mixed' Errors (True Label = Mixed, Predicted != Mixed):")
    
    error_indices = [i for i, (true, pred) in enumerate(zip(val_labels, y_pred)) if true == mixed_idx and pred != mixed_idx]
    
    error_probs = y_prob[error_indices]
    error_preds = y_pred[error_indices]
    
    errors_with_conf = []
    
    # We need indices relative to the error list
    for i, original_idx in enumerate(error_indices):
        pred_label = error_preds[i]
        confidence = error_probs[i][pred_label]
        errors_with_conf.append({
            "text": val_texts[original_idx],
            "true_label": EXPECTED_LABELS[mixed_idx],
            "predicted_label": EXPECTED_LABELS[pred_label],
            "confidence": confidence
        })
        
    # Sort by confidence descending
    errors_with_conf.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, item in enumerate(errors_with_conf[:10]):
        text_preview = item['text'][:100].replace('\n', ' ')
        print(f"{i+1}. Pred: {item['predicted_label']} ({item['confidence']:.4f}) | Text: {text_preview}...")

    print("\nDone. Improved model saved to:", OUTPUT_DIR)

    # 9. Hybrid Inference Evaluation
    print("\n" + "="*40)
    print("HYBRID INFERENCE EVALUATION (Mixed Class Fix)")
    print("="*40)
    
    # Init Predictor
    hybrid_predictor = HybridSentimentPredictor(
        model=model, 
        tokenizer=None, # Not needed since we reuse probs
        scaler=scaler,
        pos_threshold=0.45,
        neg_threshold=0.35,
        device="cpu"
    )
    
    print(f"Converting probabilities with Hybrid Strategy...")
    print(f"Thresholds: Positive >= {hybrid_predictor.pos_threshold}, Negative >= {hybrid_predictor.neg_threshold}")
    
    # We reuse y_prob calculated in step 7 to save time
    y_pred_hybrid = hybrid_predictor.apply_hybrid_rule(y_prob)
    
    print("\nClassification Report (Hybrid Strategy):")
    report_hybrid = classification_report(val_labels, y_pred_hybrid, target_names=EXPECTED_LABELS, zero_division=0)
    print(report_hybrid)
    
    # Compare Recall
    from sklearn.metrics import recall_score
    recalls = recall_score(val_labels, y_pred_hybrid, average=None, labels=[0,1,2,3])
    print(f"Mixed Recall: {recalls[0]:.4f}")
    
    # Save Hybrid Report
    report_dict_hybrid = classification_report(val_labels, y_pred_hybrid, target_names=EXPECTED_LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict_hybrid).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report_hybrid.csv"))
    print("Hybrid classification report saved.")


if __name__ == "__main__":
    main()
