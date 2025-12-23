import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

def plot_distributions(df, columns, save_dir=None):
    """Plots count distributions for specified columns."""
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/dist_{col}.png")
        else:
            plt.show()
        plt.close()

def plot_text_length(df, text_col, save_dir=None):
    """Plots the distribution of text length."""
    df['text_length'] = df[text_col].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title('Distribution of Comment Length')
    plt.xlabel('Length (characters)')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/text_length_dist.png")
    else:
        plt.show()
    plt.close()

def generate_wordcloud(text_series, title, save_path=None):
    """Generates and saves a wordcloud."""
    text = " ".join(str(t) for t in text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_time_series(df, date_col, save_dir=None):
    """Plots comments over time."""
    df[date_col] = pd.to_datetime(df[date_col])
    daily_counts = df.set_index(date_col).resample('D').size()
    
    plt.figure(figsize=(12, 6))
    daily_counts.plot()
    plt.title('Number of Comments Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/time_series.png")
    else:
        plt.show()
    plt.close()
