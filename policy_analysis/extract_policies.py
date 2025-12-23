import pandas as pd

try:
    df = pd.read_csv('c:/Users/madhu/nlp_proj/policy_analysis/data/india_policy_comments_large_50k.csv')
    print("Unique project_names:")
    for name in df['project_name'].dropna().unique():
        print(name)
    
    print("\nUnique sources:")
    for source in df['source'].dropna().unique():
        print(source)

except Exception as e:
    print(f"Error: {e}")
