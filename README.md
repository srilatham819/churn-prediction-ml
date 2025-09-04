# Churn Prediction Project

End-to-end ML pipeline with training, FastAPI, Docker, and CI/CD.



### Local
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/train.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv --out_dir artifacts
uvicorn src.infer:app --host 0.0.0.0 --port 8000


docker build -t churn-api .
docker run -p 8000:8000 churn-api


Predict (pretty):

bash
Copy code
curl -s -X POST 'http://127.0.0.1:8000/predict?pretty=1' \
  -H "Content-Type: application/json" \
  -d '{"features":{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","PaperlessBilling":"Yes","MonthlyCharges":70.35,"TotalCharges":842.45,"InternetService":"Fiber optic","Contract":"Month-to-month","PaymentMethod":"Electronic check"}}' | python -m json.tool
Metrics (pretty):

bash
Copy code
curl -s 'http://127.0.0.1:8000/metrics?pretty=1' | python -m json.tool
Sample Metrics (from my run)
json
Copy code
{
  "roc_auc": 0.8394,
  "accuracy": 0.7928,
  "f1": 0.5668
}