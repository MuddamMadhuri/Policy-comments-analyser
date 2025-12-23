
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

def test_logic():
    print("Testing Hybrid Logic...")
    
    # helper to create prob array
    def make_prob(neg, neu, pos, mix):
        p = np.zeros(4)
        p[LABEL2ID["Negative"]] = neg
        p[LABEL2ID["Neutral"]] = neu
        p[LABEL2ID["Positive"]] = pos
        p[LABEL2ID["Mixed"]] = mix
        return p

    # Case 1: Model Trust (Step 0)
    p1 = make_prob(0.1, 0.1, 0.2, 0.6)
    res1, path1, _ = hybrid_predict(p1, "some text")
    print(f"Case 1 (High Mixed Prob): {res1} via {path1}")
    assert res1 == "Mixed"
    assert "Step0" in path1

    # Case 2: Ambivalence (Step 1.5)
    # Pos=0.35, Neg=0.32 (Both > 0.25, Diff < 0.20)
    p2 = make_prob(0.32, 0.1, 0.35, 0.23)
    res2, path2, _ = hybrid_predict(p2, "neutral text without keywords")
    print(f"Case 2 (Ambivalence): {res2} via {path2}")
    assert res2 == "Mixed"
    assert "Step1.5" in path2

    # Case 3: Contrastive (Step 1)
    p3 = make_prob(0.1, 0.1, 0.4, 0.1) # Max pol > 0.3
    res3, path3, _ = hybrid_predict(p3, "It is good but also bad")
    print(f"Case 3 (Contrastive): {res3} via {path3}")
    assert res3 == "Mixed"
    assert "Step1_" in path3

    # Case 4: Clear Positive (Step 3)
    p4 = make_prob(0.05, 0.1, 0.8, 0.05)
    res4, path4, _ = hybrid_predict(p4, "This is amazing")
    print(f"Case 4 (Clear Positive): {res4} via {path4}")
    assert res4 == "Positive"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_logic()
