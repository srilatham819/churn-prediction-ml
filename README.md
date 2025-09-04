# 📘 Churn Prediction ML API


## 📌 Overview
End-to-end machine learning project that predicts telecom customer churn and exposes a production-style **FastAPI** service. It includes data download, training, stored artifacts, `/predict` and `/metrics` endpoints, **Docker** packaging, and **CI/CD with GitHub Actions**.

---

## ✨ Features
- Gradient Boosting classifier with clean preprocessing (scikit-learn Pipeline + ColumnTransformer)
- FastAPI service:
  - `GET /` – health
  - `POST /predict` – probability + label (+ `?pretty=1`)
  - `GET /metrics` – ROC AUC, Accuracy, F1 (+ `?pretty=1`)
  - `GET /docs` – interactive Swagger UI
- Dockerized for one-command run
- GitHub Actions workflow samples (tests, Docker publish)

---

## 🛠 Tech Stack
Python 3.11 • pandas • scikit-learn • joblib • FastAPI • Uvicorn • Docker • GitHub Actions

---

## 🚀 Quickstart (Local)
```bash
# clone
git clone git@github.com:srilatham819/churn-prediction-ml.git
cd churn-prediction-ml

# env
python3 -m venv .venv
source .venv/bin/activate

# deps
pip install -r requirements.txt

# data + train
python src/download_data.py
python src/train.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv --out_dir artifacts

# serve
uvicorn src.infer:app --host 0.0.0.0 --port 8000
```
- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/

**Sample calls**
```bash
# Predict (pretty)
curl -s -X POST 'http://127.0.0.1:8000/predict?pretty=1'   -H "Content-Type: application/json"   -d '{"features":{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","PaperlessBilling":"Yes","MonthlyCharges":70.35,"TotalCharges":842.45,"InternetService":"Fiber optic","Contract":"Month-to-month","PaymentMethod":"Electronic check"}}' | python -m json.tool

# Metrics (pretty)
curl -s 'http://127.0.0.1:8000/metrics?pretty=1' | python -m json.tool
```

---

## 🐳 Quickstart (Docker, build locally)
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
# Docs: http://127.0.0.1:8000/docs
```
Mount local artifacts (optional):
```bash
docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" churn-api
```

---

## 🐙 Run via GHCR (no build needed)
If you don't want to build locally, pull the public image from **GitHub Container Registry**:
```bash
docker run -p 8000:8000 ghcr.io/srilatham819/churn-api:latest
# Docs: http://127.0.0.1:8000/docs
```
> Maintainer note: steps to publish to GHCR are in **Deployment → GHCR** below.

---

## 📡 API Examples

### `/predict` (request body)
```json
{
  "features": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 70.35,
    "TotalCharges": 842.45,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }
}
```
### `/predict` (sample response)
```json
{
  "input_features": { "...": "..." },
  "churn_probability": 0.6189,
  "prediction": "Yes",
  "threshold_used": 0.5,
  "model_info": { "algorithm": "GradientBoostingClassifier", "version": "1.0.0" }
}
```

### `/metrics` (sample response)
```json
{
  "roc_auc": 0.8394,
  "accuracy": 0.7928,
  "f1": 0.5668
}
```

---

## 🔄 CI
Minimal test workflow (see `.github/workflows/ci.yml`) runs `pytest` on each push/PR and shows a ✅ badge.

Badge (replace path if you forked):
```md
![CI](https://img.shields.io/github/actions/workflow/status/srilatham819/churn-prediction-ml/ci.yml)
```

---

## 📦 Deployment

### A) GHCR (GitHub Container Registry)
**Publish once** so others can `docker run ghcr.io/srilatham819/churn-api:latest`:

1. **Login** to GHCR (create a Personal Access Token with `write:packages`):
   ```bash
   echo "<YOUR_GH_PAT>" | docker login ghcr.io -u srilatham819 --password-stdin
   ```
2. **Tag & push**:
   ```bash
   docker tag churn-api:latest ghcr.io/srilatham819/churn-api:latest
   docker push ghcr.io/srilatham819/churn-api:latest
   ```
3. **Make public**: GitHub → Profile → **Packages** → `churn-api` → **Package settings** → **Public**.

Now anyone can run:
```bash
docker run -p 8000:8000 ghcr.io/srilatham819/churn-api:latest
```

### B) Auto-publish on each commit (GitHub Actions)
Create `.github/workflows/docker-publish.yml`:

```yaml
name: Build & Push Docker image (GHCR)

on:
  push:
    branches: [main]

jobs:
  docker:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/churn-api:latest
            ghcr.io/${{ github.repository_owner }}/churn-api:${{ github.sha }}
```

### C) Versioned releases (optional)
```bash
git tag v1.0.0 && git push origin v1.0.0
docker tag churn-api:latest ghcr.io/srilatham819/churn-api:v1.0.0
docker push ghcr.io/srilatham819/churn-api:v1.0.0
```

---

## 📁 Project Structure
```
churn-prediction-ml/
├── src/
│   ├── download_data.py
│   ├── train.py
│   └── infer.py
├── tests/
├── artifacts/
├── data/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📝 License
MIT © 2025 srilatham819
