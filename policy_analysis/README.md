# Policy Draft Comment Analysis

End-to-end analysis and model development for policy support.

## Project Structure
```
policy_analysis/
├── data/                   # Dataset
├── src/                    # Source code
│   ├── preprocessing.py    # Cleaning & Splitting
│   ├── features.py         # TF-IDF & BERT
│   ├── models.py           # Model definitions
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Metrics
│   ├── visualization.py    # Plotting
│   └── insights.py         # Insight extraction
├── api/                    # FastAPI app
├── dashboard/              # Streamlit app
├── models/                 # Saved models
├── results/                # Evaluation plots
├── insights/               # Generated insights
└── run_*.py                # Execution scripts
```

## Execution Steps

### 1. Setup Environment
1.  Install dependencies:
    ```bash
    pip install -r policy_analysis/requirements.txt
    ```

### 2. Build the Model (Required)
Since the trained model files are too large for GitHub, you must regenerate them locally using the included dataset:
```bash
python policy_analysis/train_enhanced_sentiment.py
```
*   This will train the model on `india_policy_comments_large_50k.csv`.
*   Artifacts will be saved to `policy_analysis/models_improved/`.

### 3. Run the Application
Once training is complete (~5-10 mins on GPU, longer on CPU), start the server:
```bash
python -m uvicorn policy_analysis.api.main:app --reload
```
Access the UI at: [http://127.0.0.1:8000/app.html](http://127.0.0.1:8000/app.html)

### Optional: Run Dashboard
```bash
streamlit run policy_analysis/dashboard/app.py
```

3.  **Insights**:
    ```bash
    python policy_analysis/run_insights.py
    ```
    Outputs to `policy_analysis/insights/`.

4.  **API**:
    ```bash
    uvicorn policy_analysis.api.main:app --reload
    ```

5.  **Dashboard**:
    ```bash
    streamlit run policy_analysis/dashboard/app.py
    ```

## Models
-   **Sentiment**: Logistic Regression (TF-IDF)
-   **Stance**: Logistic Regression (TF-IDF)
-   **Topic**: Logistic Regression (TF-IDF)
*(BERT code is available in `src/train.py` but not enabled by default for speed)*
