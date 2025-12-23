
import sys
import os
import numpy as np

# Add the directory to path to import the module
sys.path.append(r'c:\Users\madhu\nlp_proj\policy_analysis')

try:
    from run_hybrid_inference import hybrid_predict, LABEL2ID
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

def run_tests():
    print("Testing 15 Edge Code Cases...")

    # helper to create prob array (approximate probabilities for testing overrides)
    def make_prob(neg, neu, pos, mix):
        p = np.zeros(4)
        p[LABEL2ID["Negative"]] = neg
        p[LABEL2ID["Neutral"]] = neu
        p[LABEL2ID["Positive"]] = pos
        p[LABEL2ID["Mixed"]] = mix
        return p
    
    # helper for assertions
    def check(name, text, probs, expected_label, expected_path_part=None):
        res, path, _ = hybrid_predict(probs, text)
        status = "PASS" if res == expected_label else f"FAIL (Got {res}, Exp {expected_label})"
        if expected_path_part and expected_path_part not in path:
            status = f"FAIL (Path mismatch. Got {path}, Expected *{expected_path_part}*)"
        print(f"[{status}] {name}: {res} (Path: {path})")
        return status == "PASS"

    results = []

    # 1. Contradictory (True Mixed)
    # Model often sees this as Neutral or dominant class. We force Mixed.
    p1 = make_prob(0.3, 0.4, 0.2, 0.1) 
    results.append(check("1. Contradiction", "The policy is progressive and much needed, but the implementation timeline is unrealistic and poorly planned.", p1, "Mixed", "Contrastive"))

    # 2. Neutral Language with Strong Policy Implication
    # "increases compliance costs" -> Negative/Mixed imply
    p2 = make_prob(0.2, 0.7, 0.1, 0.0) 
    results.append(check("2. Implicit Policy Concern", "This policy increases compliance costs for small businesses.", p2, "Negative", "Governance_Concern")) # Or Mixed

    # 3. Sarcasm & Irony
    # "Great... gift to bureaucracy" -> Negative
    p3 = make_prob(0.1, 0.2, 0.6, 0.1) 
    results.append(check("3. Sarcasm", "Great policy—another gift to bureaucracy.", p3, "Negative", "Sarcasm")) 

    # 4. Extremely Short Comments
    # "Good." -> Low Info or Positive. User asked for "Low Information" tag or handling. 
    # Let's assume we map very short to "Neutral" or keep as is if strong.
    # If "Good." comes with high positive prob, keep positive. If vague "Needs work.", maybe mixed/neutral.
    # Scenario: Vague short
    p4 = make_prob(0.3, 0.4, 0.3, 0.0)
    results.append(check("4. Short/Vague", "Needs work.", p4, "Neutral", "LowInfo")) 

    # 5. Long Multi-Issue
    # "support... but funding... and ignores..." -> Mixed
    p5 = make_prob(0.3, 0.2, 0.4, 0.1)
    results.append(check("5. Multi-Issue", "I support the healthcare expansion, but the funding model is unclear, and the rollout ignores rural areas.", p5, "Mixed", "Contrastive"))

    # 6. Layman Language
    # "Money part is confusing" -> Negative/Mixed
    p6 = make_prob(0.2, 0.6, 0.2, 0.0)
    results.append(check("6. Layman Language", "Money part is confusing.", p6, "Negative", "Governance_Concern"))

    # 7. Implicit Negation
    # "hardly addresses" -> Negative
    p7 = make_prob(0.2, 0.7, 0.1, 0.0)
    results.append(check("7. Implicit Negation", "The policy hardly addresses urban housing.", p7, "Negative", "Negation"))

    # 8. Conditional Support
    # "good if..." -> Mixed (User suggested Bias toward Mixed)
    p8 = make_prob(0.1, 0.2, 0.6, 0.1)
    results.append(check("8. Conditional Support", "This policy is good if proper monitoring mechanisms are added.", p8, "Mixed", "Conditional"))

    # 9. Politically Sensitive / Strong Opposition
    # "centralizes authority" -> Negative/Mixed (Ideological)
    p9 = make_prob(0.2, 0.7, 0.1, 0.0)
    results.append(check("9. Political Sensitivity", "This policy centralizes authority and reduces state autonomy.", p9, "Negative", "Governance_Concern"))

    # 10. Repetitive/Campaign (Simulated by content)
    # We can't dedup here (single text), but treat strong oppose words as Negative
    p10 = make_prob(0.8, 0.1, 0.1, 0.0)
    results.append(check("10. Campaign Language", "We strongly oppose this anti-people policy.", p10, "Negative"))

    # 11. Hinglish/Mixed Language
    # "logic is good but implementation bohot weak hai" -> Mixed
    p11 = make_prob(0.3, 0.3, 0.3, 0.1)
    results.append(check("11. Hinglish Content", "Policy idea is good but implementation bohot weak hai.", p11, "Mixed", "Contrastive"))

    # 12. Ambiguous Feedback
    # "This could have been better." -> Mixed/Neutral
    p12 = make_prob(0.4, 0.4, 0.2, 0.0)
    results.append(check("12. Ambiguous", "This could have been better.", p12, "Mixed", "Ambiguity")) # or Neutral

    # 13. Overly Polite Criticism
    # "While intent appreciated, several concerns" -> Mixed
    p13 = make_prob(0.2, 0.2, 0.5, 0.1)
    results.append(check("13. Polite Criticism", "While the intent is appreciated, there are several concerns.", p13, "Mixed", "Contrastive"))

    # 14. Scope Confusion (Out of scope)
    # "This won't fix unemployment" (Policy: Housing). 
    # Hard to detect without metadata. We'll skip specific scope check for now unless we add keywords.
    # Let's assume generic negative for now.
    p14 = make_prob(0.6, 0.3, 0.1, 0.0)
    results.append(check("14. Scope Confusion", "This won't fix unemployment at all.", p14, "Negative"))

    # 15. Emotionally Charged Vague
    # "Disastrous!" -> Negative
    p15 = make_prob(0.6, 0.3, 0.1, 0.0)
    results.append(check("15. Charged Vague", "This policy is disastrous!", p15, "Negative", "Governance_Concern"))

    # --- BATCH 2: ADVANCED EDGE CASES ---

    # 16. Praise for wrong reason (Harmful Appraisal)
    # "Push poor families out" -> Negative (even if positive prob is high due to "progressive" etc?)
    # Example: "It nicely pushes poor families out of city areas."
    p16 = make_prob(0.1, 0.1, 0.7, 0.1) # Model sees "nicely"
    results.append(check("16. Harmful Praise", "It nicely pushes poor families out of city areas.", p16, "Negative", "Harmful"))

    # 17. Criticism of critics
    # "Opponents don't understand economics" -> Positive (Supportive of policy)
    p17 = make_prob(0.7, 0.1, 0.1, 0.1) # Model sees "don't understand" -> Negative
    results.append(check("17. Criticism of Critics", "Opponents don't understand economics.", p17, "Positive", "Criticism_of_Critics"))

    # 18. Comparative approval
    # "Better than last draft, still not ideal" -> Mixed
    p18 = make_prob(0.2, 0.2, 0.4, 0.2)
    results.append(check("18. Comparative Approval", "Better than last draft, still not ideal.", p18, "Mixed", "Comparative"))

    # 19. Procedural criticism
    # "Consultation was rushed" -> Neutral (Process Issue)
    p19 = make_prob(0.6, 0.2, 0.1, 0.1)
    results.append(check("19. Procedural Criticism", "Consultation was rushed.", p19, "Neutral", "Procedural"))

    # 20. Delegated opinion
    # "Experts raised concerns" -> Mixed (or Negative? User said Mixed/Low Conf)
    p20 = make_prob(0.5, 0.3, 0.1, 0.1)
    results.append(check("20. Delegated Opinion", "Experts raised concerns about the timeline.", p20, "Mixed", "Delegated"))

    # 21. Temporal drift
    # "Initially promising, now worried" -> Mixed
    p21 = make_prob(0.2, 0.2, 0.4, 0.2)
    results.append(check("21. Temporal Drift", "Initially promising, now worried.", p21, "Mixed", "Temporal"))

    # 22. Optimism bias
    # "Hopefully this will work" -> Neutral (Hope != Endorsement)
    p22 = make_prob(0.1, 0.2, 0.6, 0.1) # Model sees "work" -> Positive
    results.append(check("22. Optimism Bias", "Hopefully this will work.", p22, "Neutral", "Optimism"))

    # 23. Policy capture language
    # "Aligns with large developer interests" -> Negative
    p23 = make_prob(0.2, 0.6, 0.2, 0.0)
    results.append(check("23. Policy Capture", "Aligns with large developer interests.", p23, "Negative", "Capture"))

    # 24. Fear-based support
    # "We need this or chaos will follow" -> Mixed
    p24 = make_prob(0.4, 0.1, 0.4, 0.1)
    results.append(check("24. Fear-based Support", "We need this or chaos will follow.", p24, "Mixed", "Fear"))

    # 25. Conditional opposition
    # "If enforcement stays weak, it will fail" -> Mixed
    p25 = make_prob(0.7, 0.1, 0.1, 0.1)
    results.append(check("25. Conditional Opposition", "If enforcement stays weak, it will fail.", p25, "Mixed", "Conditional"))

    # 26. Moral framing
    # "Housing is a basic human right" -> Neutral (Normative)
    p26 = make_prob(0.1, 0.4, 0.4, 0.1)
    results.append(check("26. Moral Framing", "Housing is a basic human right.", p26, "Neutral", "Moral"))

    # 27. Statistical skepticism
    # "Numbers seem selectively chosen" -> Negative
    p27 = make_prob(0.2, 0.5, 0.2, 0.1)
    results.append(check("27. Stat Skepticism", "Numbers seem selectively chosen.", p27, "Negative", "Skepticism"))

    # 28. Hypothetical scenario
    # "If applied nationwide, rents might rise" -> Mixed
    p28 = make_prob(0.5, 0.3, 0.1, 0.1)
    results.append(check("28. Hypothetical", "If applied nationwide, rents might rise.", p28, "Mixed", "Hypothetical"))

    # 29. Silent Agreement
    # "Follows global best practices" -> Positive
    p29 = make_prob(0.2, 0.5, 0.2, 0.1) # Model might be Neutral
    results.append(check("29. Silent Agreement", "Follows global best practices.", p29, "Positive", "Silent_Agreement"))

    # 30. Institutional politeness mask
    # "Requires substantial reconsideration" -> Negative
    p30 = make_prob(0.2, 0.5, 0.2, 0.1) # Soft words
    results.append(check("30. Politeness Mask", "Requires substantial reconsideration.", p30, "Negative", "Politeness"))

    passed = results.count(True)
    total = len(results)
    print(f"\nFinal Result: {passed}/{total} Passed")

if __name__ == "__main__":
    run_tests()
