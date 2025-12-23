
# Config
LABELS = ["Negative", "Neutral", "Positive", "Mixed"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Keywords
# Strict contrastive cues
CONTRAST_KEYWORDS = ["but", "however", "although", "yet", "nevertheless", "though", "while", "unlike", "despite"]

# Conditional Support (Bias to Mixed)
CONDITIONAL_KEYWORDS = ["if", "provided that", "unless", "assuming", "subject to", "only when", "needs to"]

# Model Trust Thresholds
PROB_MIXED_THRESHOLD = 0.50

# Domain-Specific Analytical Approvals (Policy Positive Lexicon)
POLICY_POSITIVE_TERMS = [
  "aligns with", "aligned with", "well designed", "appropriate", "reasonable", "balanced",
  "commendable", "welcome step", "step in the right direction", "effective framework",
  "good initiative", "strong foundation", "international best practices", "beneficial",
  "crucial", "essential", "supportive", "timely", "comprehensive", "robust",
  "forward-looking", "pragmatic approach", "valuable addition", "much needed",
  "agree with", "concur", "endorse", "support", "fully support", "in favor of",
  "positive step", "progressive", "innovative", "landmark", "milestone",
  "appreciated", "encourage", "recommend", "ideal", "optimal", "sound",
  "logical", "rational", "improves", "enhances", "strengthens", "facilitates",
  "promotes", "fosters", "constructive", "promising", "useful", "helpful",
  "practical", "feasible", "viable", "clear", "transparent", "well-structured",
  "well-defined", "proper", "solid", "meaningful", "important"
]

def hybrid_predict(p, text):
    """
    Core Inference Logic (Robust Hybrid V2).
    p: list/array of probabilities [Neg, Neu, Pos, Mix]
    text: raw text string
    Returns: (decision_label, path_description, has_contrast)
    """
    # STEP -1: INVALID INPUT CHECK (Before checking probabilities)
    stripped_text = text.strip()
    if len(stripped_text) < 4:
        return "Invalid", "Step-1_Too_Short", False
    
    import re
    if not re.search(r'[a-zA-Z]{2,}', stripped_text):
        return "Invalid", "Step-1_No_Letters", False

    # Unpack probabilities safely
    try:
        p_neg = p[LABEL2ID["Negative"]]
        p_neu = p[LABEL2ID["Neutral"]]
        p_pos = p[LABEL2ID["Positive"]]
    except IndexError:
        # Fallback if model behaves unexpectedly
        return "Error", "Prob_Index_Error", False

    # Safe Mixed Access (Handle 3-class models)
    try:
        p_mix = p[LABEL2ID["Mixed"]]
    except (IndexError, KeyError):
        p_mix = 0.0
    
    text_lower = text.lower()
    
    # Helper for word boundary matching
    def check_keywords(txt, kw_list):
        import re
        pattern = r"\b(" + "|".join(map(re.escape, kw_list)) + r")\b"
        return bool(re.search(pattern, txt))

    has_contrast = check_keywords(text_lower, CONTRAST_KEYWORDS)
    has_condition = check_keywords(text_lower, CONDITIONAL_KEYWORDS)
    
    # Calculate Derived Metrics
    max_pol = max(p_neg, p_pos)
    
    # STEP 0: MODEL TRUST (Explicit Prediction)
    if p_mix >= PROB_MIXED_THRESHOLD:
        return "Mixed", "Step0_Model_High_Prob", has_contrast

    # STEP 1: CONDITIONAL SUPPORT (Rule-Based Mixed)
    if has_condition:
        return "Mixed", "Step1_Conditional_Mixed", True

    # STEP 1.1: MIXED: QUALIFIED SENTIMENT
    if (has_contrast and max_pol >= 0.25):
        return "Mixed", "Step1.1_Contrastive_Mixed", has_contrast

    # STEP 1.6: AMBIVALENCE check (Implicit Conflict)
    if p_pos > 0.20 and p_neg > 0.20 and abs(p_pos - p_neg) < 0.25:
        return "Mixed", "Step1.6_Ambivalence_Conflict", has_contrast

    # STEP 2: NEUTRAL AS PURE ABSENCE
    if max_pol < 0.25:
        return "Neutral", "Step2_Neutral_Absence", has_contrast

    # STEP 3: POLARITY RESOLUTION
    if p_neg >= 0.35 and p_neg >= p_pos:
        return "Negative", "Step3_Polarity_Negative", has_contrast
    
    if p_pos >= 0.35:
        return "Positive", "Step3_Polarity_Positive", has_contrast

    # STEP 3.5: ANALYTICAL POSITIVE BOOST (Catch Weak Signals)
    if any(term in text_lower for term in POLICY_POSITIVE_TERMS):
        return "Positive", "Step3.5_Analytical_Boost", has_contrast

    # STEP 4: FINAL FALLBACK (NO NEUTRAL)
    if p_neg >= p_pos:
        return "Negative", "Step4_Fallback_Negative", has_contrast
    else:
        return "Positive", "Step4_Fallback_Positive", has_contrast
