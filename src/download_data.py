# src/download_data.py
import pandas as pd
from pathlib import Path

# Public IBM Telco Customer Churn dataset
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL)
    df.to_csv(OUT, index=False)
    print(f"✅ Saved dataset to {OUT}")

if __name__ == "__main__":
    main()
