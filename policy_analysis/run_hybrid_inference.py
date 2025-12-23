
import os
import torch
import re
import pandas as pd
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_improved/sentiment_model_v2")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.csv") 
RAW_DATA_PATH = os.path.join(BASE_DIR, "../india_policy_comments_large_50k.csv")
OUTPUT_CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix_hybrid.csv")
EXPLAINABILITY_PATH = os.path.join(MODEL_DIR, "inference_explainability.csv")
TEMPERATURE_PATH = os.path.join(MODEL_DIR, "temperature.txt")

# Config
LABELS = ["Negative", "Neutral", "Positive", "Mixed"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# Keywords
# Strict contrastive cues
CONTRAST_KEYWORDS = ["but", "however", "although", "yet", "nevertheless", "though", "while", "unlike", "despite"]

# Conditional Support (Bias to Mixed)
CONDITIONAL_KEYWORDS = ["if", "provided that", "unless", "assuming", "subject to", "only when", "needs to"]

# Implicit Negation (Neutral -> Negative)
NEGATION_ADVERBS = ["hardly", "barely", "scarcely", "fails to", "lack of", "lacks", "absent"]

# Governance/Political Sensitivity (Neutral -> Negative/Mixed)
GOVERNANCE_CRITICISM = [
    "centralizes", "autonomy", "compliance", "bureaucracy", "red tape", "burden", 
    "cost", "unrealistic", "poorly planned", "confusing", "unclear", "disastrous", "harmful"
]

# Sarcasm Cues (Positive words potentially used ironically)
SARCASM_OPRAISE = ["great", "good", "amazing", "wonderful", "gift"]
SARCASM_TARGETS = ["bureaucracy", "red tape", "delay", "tax", "corruption"]

# --- BATCH 2 LEXICONS ---
# 16. Harmful Appraisal
HARMFUL_TERMS = ["push poor", "displace", "exclude", "marginalize", "pushes poor"]

# 17. Criticism of Critics (Positive)
CRITICS_TERMS = ["opponents", "critics", "naysayers"]
CRITICS_ERROR = ["wrong", "don't understand", "misunderstand", "incorrect", "false"]

# 18. Comparative Approval (Mixed)
COMPARATIVE_TERMS = ["better than", "improvement over", "improved but", "still not ideal"]

# 19. Procedural (Neutral)
PROCEDURAL_TERMS = ["consultation", "rushed", "timeline", "drafting", "process"]

# 20. Delegated (Mixed)
DELEGATED_TERMS = ["experts raised", "concerns raised", "experts say"]

# 21. Temporal Drift (Mixed)
TEMPORAL_START = ["initially", "at first", "started as"]
TEMPORAL_END = ["now", "currently", "turned into"]

# 22. Optimism (Neutral)
OPTIMISM_TERMS = ["hopefully", "hope", "fingers crossed"]

# 23. Policy Capture (Negative)
CAPTURE_TERMS = ["developer interests", "corporate lobby", "rich", "elites", "big infra"]

# 24. Fear (Mixed)
FEAR_TERMS = ["chaos", "disaster", "collapse", "crisis"]

# 25. Conditional Opposition (Mixed) - reusing CONDITIONAL_KEYWORDS

# 26. Moral Framing (Neutral)
MORAL_TERMS = ["human right", "moral duty", "ethical", "basic right"]

# 27. Stat Skepticism (Negative)
SKEPTICISM_TERMS = ["selectively chosen", "misleading numbers", "fake data", "skewed", "manipulated"]

# 28. Hypothetical (Mixed)
HYPOTHETICAL_TERMS = ["if applied", "might rise", "could lead to", "potential risk"]

# 29. Silent Agreement (Positive)
SILENT_AGREE_TERMS = ["best practices", "global standard", "international standard", "follows global"]

# 30. Politeness Mask (Negative)
POLITENESS_MASK = ["substantial reconsideration", "requires review", "not fully aligned", "major revision"]

# Domain-Specific Analytical Approvals (Policy Positive Lexicon)

# Domain-Specific Analytical Approvals (Policy Positive Lexicon)
POLICY_POSITIVE_TERMS = [
  "aligns with",
  "aligned with",
  "well designed",
  "appropriate",
  "reasonable",
  "balanced",
  "commendable",
  "welcome step",
  "step in the right direction",
  "effective framework",
  "good initiative",
  "strong foundation",
  "international best practices",
  "beneficial",
  "crucial", 
  "essential",
  "supportive",
  "timely",
  "comprehensive",
  "robust",
  "forward-looking",
  "pragmatic approach",
  "valuable addition",
  "much needed",
  "agree with",
  "concur",
  "endorse",
  "support",
  "fully support",
  "in favor of",
  "positive step",
  "progressive",
  "innovative",
  "landmark",
  "milestone",
  "appreciated",
  "encourage",
  "recommend",
  "ideal",
  "optimal",
  "sound",
  "logical",
  "rational",
  "improves",
  "enhances",
  "strengthens",
  "facilitates",
  "promotes",
  "fosters",
  "constructive",
  "promising",
  "useful",
  "helpful",
  "practical",
  "feasible",
  "viable",
  "clear",
  "transparent",
  "well-structured",
  "well-defined",
  "proper",
  "solid",
  "meaningful",
  "important"
]


        

# Valid Punctuation for sentence splitting
SENTENCE_SPLIT_REGEX = r'(?<=[.!?])\s+'

def split_sentences(text):
    """Splits text into sentences."""
    sentences = re.split(SENTENCE_SPLIT_REGEX, text)
    return [s.strip() for s in sentences if s.strip()]

def aggregate_sentiments(sent_labels):
    """
    Aggregates a list of sentence labels into a document label.
    """
    unique_labels = set(sent_labels)
    has_pos = "Positive" in unique_labels
    has_neg = "Negative" in unique_labels
    
    # 1. Co-existence = Mixed
    if has_pos and has_neg:
        return "Mixed", "Aggregation_PosNeg_Found"
        
    # 2. Homogeneous signals
    if has_pos:
        return "Positive", "Aggregation_Positive"
    if has_neg:
        return "Negative", "Aggregation_Negative"
        
    # 3. Default
    return "Neutral", "Aggregation_Neutral"

def hybrid_predict(p, text, aggregated_label=None, aggregated_path=None):
    # Unpack probabilities
    p_neg = p[LABEL2ID["Negative"]]
    p_pos = p[LABEL2ID["Positive"]]
    p_mix = p[LABEL2ID["Mixed"]]
    p_neu = p[LABEL2ID["Neutral"]]
    
    text_lower = text.lower()
    
    # Helper for word boundary matching
    def check_keywords(txt, kw_list):
        pattern = r"\b(" + "|".join(map(re.escape, kw_list)) + r")\b"
        return bool(re.search(pattern, txt))

    has_contrast = check_keywords(text_lower, CONTRAST_KEYWORDS)
    has_condition = check_keywords(text_lower, CONDITIONAL_KEYWORDS)
    has_negation = check_keywords(text_lower, NEGATION_ADVERBS)
    
    # Batch 2 Checks
    has_harm = check_keywords(text_lower, HARMFUL_TERMS)
    has_critics = check_keywords(text_lower, CRITICS_TERMS)
    has_error = check_keywords(text_lower, CRITICS_ERROR)
    has_comparative = check_keywords(text_lower, COMPARATIVE_TERMS)
    has_procedural = check_keywords(text_lower, PROCEDURAL_TERMS)
    has_delegated = check_keywords(text_lower, DELEGATED_TERMS)
    has_temp_start = check_keywords(text_lower, TEMPORAL_START)
    has_temp_end = check_keywords(text_lower, TEMPORAL_END)
    has_optimism = check_keywords(text_lower, OPTIMISM_TERMS)
    has_capture = check_keywords(text_lower, CAPTURE_TERMS)
    has_fear = check_keywords(text_lower, FEAR_TERMS)
    has_moral = check_keywords(text_lower, MORAL_TERMS)
    has_skepticism = check_keywords(text_lower, SKEPTICISM_TERMS)
    has_hypothetical = check_keywords(text_lower, HYPOTHETICAL_TERMS)
    has_silent = check_keywords(text_lower, SILENT_AGREE_TERMS)
    has_polite = check_keywords(text_lower, POLITENESS_MASK)

    # Calculate Derived Metrics
    max_pol = max(p_neg, p_pos)
    
    # STEP -1: AGGREGATION CHECK (Integration Point)
    # If Sentence Aggregation found "Mixed", we prioritize it unless Doc-Level probability is strongly Non-Mixed.
    if aggregated_label == "Mixed":
         # Safety: Only trust aggregation if Doc-Level isn't super confident in a single polarity
         if max_pol < 0.85: 
             return "Mixed", f"Step-1_{aggregated_path}", has_contrast

    # STEP 0: SHORT COMMENT HANDLING (Edge Case 4)
    # If very short (< 3 words) and no strong signal, flag as Neutral/LowInfo
    if len(text.split()) < 3 and max_pol < 0.6:
         return "Neutral", "Step0_Short_LowInfo", False

    # STEP 0.1: OPTIMISM BIAS (Edge Case 22)
    # Hope != Endorsement. 
    if has_optimism and max_pol < 0.7: # Only override if not extremely strong
         return "Neutral", "Step0.1_Optimism_Neutral", False

    # STEP 0.5: MODEL TRUST (Explicit Prediction)
    if p_mix >= 0.50:
        return "Mixed", "Step0_Model_High_Prob", has_contrast

    # STEP 1.4: HYPOTHETICAL / FEAR (Edge Case 24, 28)
    if has_hypothetical or has_fear:
        return "Mixed", "Step1.4_Hypothetical_Fear", True

    # STEP 1: CONDITIONAL SUPPORT (Edge Case 8, 25)
    # "Good if...", "Fail if..." -> Mixed
    # Check for conditional keywords. 
    if has_condition:
        return "Mixed", "Step1_Conditional_Mixed", True

    # STEP 1.1: MIXED: QUALIFIED SENTIMENT (Edge Case 1, 5, 11)
    if (has_contrast and max_pol >= 0.25):
        return "Mixed", "Step1.1_Contrastive_Mixed", has_contrast

    # STEP 1.2: TEMPORAL DRIFT (Edge Case 21)
    if has_temp_start and has_temp_end:
        return "Mixed", "Step1.2_Temporal_Drift", True
        
    # STEP 1.3: COMPARATIVE APPROVAL (Edge Case 18)
    if has_comparative:
        return "Mixed", "Step1.3_Comparative_Approval", True
        
    # STEP 1.5: DELEGATED OPINION (Edge Case 20)
    if has_delegated:
         return "Mixed", "Step1.5_Delegated_Opinion", True

    # STEP 1.6: AMBIVALENCE check (Implicit Conflict) (Edge Case 1)
    if p_pos > 0.20 and p_neg > 0.20 and abs(p_pos - p_neg) < 0.25:
        return "Mixed", "Step1.6_Ambivalence_Conflict", has_contrast

    # STEP 2.0: SARCASM DETECTION (Edge Case 3)
    has_sarcasm_praise = check_keywords(text_lower, SARCASM_OPRAISE)
    has_sarcasm_target = check_keywords(text_lower, SARCASM_TARGETS)
    if has_sarcasm_praise and has_sarcasm_target:
        return "Negative", "Step2_Sarcasm_Override", True

    # STEP 2.1: CRITICISM OF CRITICS (Edge Case 17) -> Positive
    if has_critics and has_error:
        return "Positive", "Step2.1_Criticism_of_Critics", True

    # STEP 2.2: SILENT AGREEMENT (Edge Case 29) -> Positive
    if has_silent:
        return "Positive", "Step2.2_Silent_Agreement", False

    # STEP 2.3: HARMFUL APPRAISAL (Edge Case 16) -> Negative
    if has_harm:
        return "Negative", "Step2.3_Harmful_Appraisal", False

    # STEP 2.4: POLICY CAPTURE (Edge Case 23) -> Negative
    if has_capture:
        return "Negative", "Step2.4_Policy_Capture", False

    # STEP 2.5: STAT SKEPTICISM (Edge Case 27) -> Negative
    if has_skepticism:
        return "Negative", "Step2.5_Stat_Skepticism", False

    # STEP 2.6: POLITENESS MASK (Edge Case 30) -> Negative
    if has_polite:
        return "Negative", "Step2.6_Politeness_Mask", False

    # STEP 2.7: POLITICALLY SENSITIVE / IMPLICIT CRITICISM (Edge Case 2, 6, 9, 13)
    has_gov_crit = check_keywords(text_lower, GOVERNANCE_CRITICISM)
    if has_gov_crit:
         if p_pos < 0.4:
             return "Negative", "Step2.7_Governance_Concern", False

    # STEP 2.8: IMPLICIT NEGATION (Edge Case 7)
    if has_negation and p_pos < 0.4:
        return "Negative", "Step2.8_Implicit_Negation", False

    # STEP 2.9: AMBIGUITY CHECK (Edge Case 12)
    if "could be better" in text_lower or "could have been better" in text_lower:
        return "Mixed", "Step2.9_Ambiguity_Mixed", False

    # STEP 3: PROCEDURAL / MORAL / NEUTRAL (Edge Case 19, 26)
    if has_procedural:
        return "Neutral", "Step3_Procedural_Neutral", False
    if has_moral:
        return "Neutral", "Step3_Moral_Neutral", False

    # STEP 3.1: NEUTRAL AS PURE ABSENCE
    if max_pol < 0.25:
        return "Neutral", "Step3.1_Neutral_Absence", has_contrast

    # STEP 4: POLARITY RESOLUTION
    if p_neg >= 0.35 and p_neg >= p_pos:
        return "Negative", "Step4_Polarity_Negative", has_contrast
    
    if p_pos >= 0.35:
        return "Positive", "Step4_Polarity_Positive", has_contrast

    # STEP 4.5: ANALYTICAL POSITIVE BOOST (Catch Weak Signals)
    # If explicit praise terms are present, override "Weak Fallback Negative"
    if any(term in text.lower() for term in POLICY_POSITIVE_TERMS):
        # Safety: Only if NOT a Strong Negative (already handled in Step 4)
        return "Positive", "Step4.5_Analytical_Boost", has_contrast

    # STEP 5: FINAL FALLBACK (NO NEUTRAL)
    if p_neg >= p_pos:
        return "Negative", "Step5_Fallback_Negative", has_contrast
    else:
        return "Positive", "Step5_Fallback_Positive", has_contrast

class StructuralHybridClassifier:
    def __init__(self, model_dir):
        print(f"Loading model from {model_dir}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(DEVICE)
        self.model.eval()
        
        # Load Temperature (Calibration)
        if os.path.exists(TEMPERATURE_PATH):
            with open(TEMPERATURE_PATH, "r") as f:
                self.temperature = float(f.read().strip())
            print(f"Loaded Temperature: {self.temperature}")
        else:
            self.temperature = 1.0
            print("Warning: Temperature not found. Using 1.0 (Uncalibrated)")

        print(f"Model ID2LABEL: {self.model.config.id2label}")


    def predict_batch(self, texts):
        """Helper to predict simple labels for a batch of texts (sentence-level uses)."""
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # No temp scaling needed for simple argmax, but consistent to use it if we want probs later
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
        return [ID2LABEL[p] for p in preds]

    def predict_structural(self, texts):
        # 1. Document Level Inference
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply Temperature Scaling (Standard Calibration)
            scaled_logits = logits / self.temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1).cpu().numpy()
            
        final_labels = []
        raw_labels = []
        explanations = []
        
        # 2. Sentence Level Inference (Batch mode for efficiency)
        # We need to process all sentences from all docs. 
        # Strategy: Flatten all sentences, predict, then reconstruct.
        all_sentences = []
        doc_sentence_map = [] # List of (start_index, length) for each doc
        
        current_idx = 0
        for text in texts:
            sents = split_sentences(text)
            all_sentences.extend(sents)
            doc_sentence_map.append((current_idx, len(sents)))
            current_idx += len(sents)
            
        # Run inference on all sentences (chunked if massive, but assuming batch fits for now or let internal batching handle it)
        # For simplicity in this step, we assume they fit or we process in chunks. 
        # Let's do a simple chunking here to be safe.
        sent_labels_flat = []
        chunk_size = 64
        for i in range(0, len(all_sentences), chunk_size):
            chunk = all_sentences[i:i+chunk_size]
            if not chunk: continue
            chunk_labels = self.predict_batch(chunk)
            sent_labels_flat.extend(chunk_labels)
            
        # Reconstruct and Aggregate
        doc_aggregated_results = []
        for start, length in doc_sentence_map:
            if length == 0:
                doc_aggregated_results.append(("Neutral", "Empty_Text"))
                continue
            
            doc_sents_labels = sent_labels_flat[start : start+length]
            agg_lbl, agg_path = aggregate_sentiments(doc_sents_labels)
            doc_aggregated_results.append((agg_lbl, agg_path))
        
        # 3. Combine Logic
        for i, text in enumerate(texts):
            p = probs[i]
            # Record Raw Prediction (Argmax) for reference
            raw_labels.append(ID2LABEL[np.argmax(p)])
            
            # Retrieve Aggregated Signal
            agg_lbl, agg_path = doc_aggregated_results[i]
            
            # Use Single Function for Decision
            decision, path, has_contrast = hybrid_predict(p, text, aggregated_label=agg_lbl, aggregated_path=agg_path)
                    
            final_labels.append(decision)
            explanations.append({
                "text_snippet": text[:50],
                "raw_probs": p,
                "contrast": has_contrast,
                "final_label": decision,
                "decision_path": path,
                "aggregated_label": agg_lbl,
                "aggregated_path": agg_path
            })
            
        return raw_labels, final_labels, explanations

def run_pipeline():
    print(">>> Initializing Structural Classifier (Strict Negative-First Logic)...")
    try:
        classifier = StructuralHybridClassifier(MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Data
    if os.path.exists(TEST_DATA_PATH):
        print(f"Loading test data from {TEST_DATA_PATH}...")
        df = pd.read_csv(TEST_DATA_PATH)
    else:
        print(f"Test data not found. Loading sample from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH).sample(2000, random_state=42)
    
    # Filter
    df = df[df['sentiment'].isin(LABELS)].copy()
    texts = df['comment_text'].tolist()
    true_labels = df['sentiment'].tolist()
    
    print(f"Running Inference on {len(df)} samples...")
    batch_size = 32
    all_raw = []
    all_hybrid = []
    all_explanations = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        raw_batch, hybrid_batch, expl_batch = classifier.predict_structural(batch_texts)
        all_raw.extend(raw_batch)
        all_hybrid.extend(hybrid_batch)
        all_explanations.extend(expl_batch)
        
    # Stats
    print("\n>>> Classification Report")
    print("--- RAW BERT (Temperature Scaled) ---")
    print(classification_report(true_labels, all_raw, target_names=LABELS, labels=LABELS))
    print("--- STARVATION-PROOF LOGIC (Enhanced Mixed Restoration) ---")
    print(classification_report(true_labels, all_hybrid, target_names=LABELS, labels=LABELS))
    
    # Artifacts
    cm = confusion_matrix(true_labels, all_hybrid, labels=LABELS)
    pd.DataFrame(cm, index=[f"True_{l}" for l in LABELS], columns=[f"Pred_{l}" for l in LABELS]).to_csv(OUTPUT_CM_PATH)
    print(f"Confusion Matrix saved to {OUTPUT_CM_PATH}")
    
    expl_df = pd.DataFrame(all_explanations)
    expl_df['true_label'] = true_labels
    expl_df.to_csv(EXPLAINABILITY_PATH, index=False)
    print(f"Explainability Log saved to {EXPLAINABILITY_PATH}")

if __name__ == "__main__":
    run_pipeline()
