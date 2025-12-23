# Deployment Strategy

## 1. Local Deployment
To run the application locally:

### API (FastAPI)
```bash
uvicorn policy_analysis.api.main:app --reload
```
Access docs at: `http://localhost:8000/docs`

### Dashboard (Streamlit)
```bash
streamlit run policy_analysis/dashboard/app.py
```
Access dashboard at: `http://localhost:8501`

## 2. Cloud Deployment (Render/Railway)
These platforms are easiest for Python apps.

### Steps:
1.  **Push to GitHub**: Ensure `requirements.txt` is present.
2.  **Create Web Service**: Connect repository.
3.  **Build Command**: `pip install -r policy_analysis/requirements.txt`
4.  **Start Command**: `uvicorn policy_analysis.api.main:app --host 0.0.0.0 --port $PORT`

For Streamlit on Render:
-   Start Command: `streamlit run policy_analysis/dashboard/app.py --server.port $PORT --server.address 0.0.0.0`

## 3. AWS Deployment (EC2)
For full control and scalability.

### Steps:
1.  **Launch EC2 Instance**: Ubuntu 20.04 or Amazon Linux 2.
2.  **SSH into Instance**: `ssh -i key.pem ubuntu@<ip>`
3.  **Install Dependencies**:
    ```bash
    sudo apt update
    sudo apt install python3-pip nginx
    git clone <repo_url>
    cd <repo_dir>
    pip3 install -r policy_analysis/requirements.txt
    ```
4.  **Run with Gunicorn/Uvicorn**:
    ```bash
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker policy_analysis.api.main:app
    ```
5.  **Configure Nginx**: Reverse proxy to port 8000.
