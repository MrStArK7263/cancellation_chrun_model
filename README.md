# Customer Churn – End-to-End ML Project

## Files
| File | Purpose |
|---|---|
| `churn_model.ipynb` | Full notebook: data understanding -> cleaning -> EDA -> correlation -> features -> selection -> models -> tuning -> save |
| `preprocessing.py` | Feature engineering + pipeline builder (used by notebook AND API) |
| `main.py` | FastAPI backend (`/health`, `/model-info`, `/predict`) |
| `frontend/app.py` | Streamlit UI |
| `model/` | Saved pipeline (`.joblib`) + `model_metadata.json` (versioning) |
| `Dockerfile`, `requirements.txt`, `.env.example`, `render.yaml` | Deployment |

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload                       # API  -> http://127.0.0.1:8000/docs
pip install -r frontend/requirements.txt
streamlit run frontend/app.py                   # UI   -> http://localhost:8501
```

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Deploy
1. Push this folder to GitHub.
2. Render -> New Web Service -> connect repo -> Runtime **Docker** (uses `Dockerfile`; `render.yaml` has the settings). Health check path: `/health`.
3. Streamlit Community Cloud -> New app -> main file `frontend/app.py`, requirements `frontend/requirements.txt`. In *Secrets* add: `API_URL = "https://<your-service>.onrender.com"`.
4. Assignment also mentions AWS: the same Docker image can be pushed to ECR and run on App Runner / ECS.

> scikit-learn version in `requirements.txt` (1.9.1) must match the one that trained the model, otherwise loading the `.joblib` can fail.
> Render free tier sleeps after inactivity: the first request can take ~1 minute.
