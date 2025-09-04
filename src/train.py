import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender","SeniorCitizen","Partner","Dependents","PhoneService",
    "PaperlessBilling","InternetService","Contract","PaymentMethod"
]

def load_data(path: str):
    df = pd.read_csv(path)
    # clean numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # target
    y = (df["Churn"] == "Yes").astype(int)
    X = df[NUMERIC + CATEGORICAL].copy()
    return X, y

def train(data_path: str, out_dir: str):
    X, y = load_data(data_path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    preds = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "accuracy": float(accuracy_score(yte, preds)),
        "f1": float(f1_score(yte, preds)),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("✅ Saved model to", out / "model.joblib")
    print("📊 Metrics:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()
    train(args.data_path, args.out_dir)
