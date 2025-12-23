import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization import plot_distributions, plot_text_length, generate_wordcloud, plot_time_series

DATA_PATH = "policy_analysis/data/india_policy_comments_large_50k.csv"
OUTPUT_DIR = "policy_analysis/eda_output"

def run_eda():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
        return

    print(f"Dataset Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Missing values
    missing = df.isnull().sum()
    print("\nMissing Values:\n", missing[missing > 0])
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicates: {duplicates}")
    
    # Class Imbalance
    print("\nClass Distribution (Sentiment):\n", df['sentiment'].value_counts(normalize=True))
    print("\nClass Distribution (Stance):\n", df['stance'].value_counts(normalize=True))
    
    # Plots
    print("\nGenerating plots...")
    plot_distributions(df, ['sentiment', 'stance', 'topic', 'source', 'project_name'], save_dir=OUTPUT_DIR)
    plot_text_length(df, 'comment_text', save_dir=OUTPUT_DIR)
    plot_time_series(df, 'date_submitted', save_dir=OUTPUT_DIR)
    
    print("\nGenerating WordCloud...")
    generate_wordcloud(df['comment_text'], "Common Words in Comments", save_path=f"{OUTPUT_DIR}/wordcloud.png")
    
    # Insights
    print("\nTop Topics:\n", df['topic'].value_counts().head(5))
    
    # Save summary stats
    with open(f"{OUTPUT_DIR}/eda_summary.txt", "w") as f:
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Duplicates: {duplicates}\n")
        f.write("\nMissing Values:\n")
        f.write(missing.to_string())
        f.write("\n\nSentiment Distribution:\n")
        f.write(df['sentiment'].value_counts().to_string())
        f.write("\n\nStance Distribution:\n")
        f.write(df['stance'].value_counts().to_string())

    print(f"\nEDA completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda()
