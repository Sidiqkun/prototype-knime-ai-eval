"""Manual (human-built) baseline: time split, 3 models, with/without class weight, plus a deliberate
leakage scenario to show how misleading metrics become.
Usage: python src/manual_baseline.py --data data/invoices.csv
"""
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from common import FEATURES_CAT, FEATURES_NUM, TARGET, metrics

def build(model, num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("m", model)])

def main(path, test_frac, seed, outdir):
    df = pd.read_csv(path, parse_dates=["invoice_date"]).sort_values("invoice_date")
    cut = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f"train={len(train)} test={len(test)} | late rate train={train[TARGET].mean():.1%} test={test[TARGET].mean():.1%}")
    rows = []
    for leak in (False, True):
        num = FEATURES_NUM + (["days_overdue"] if leak else [])  # days_overdue = leakage
        for cw in (None, "balanced"):
            models = {
                "logreg": LogisticRegression(max_iter=1000, class_weight=cw),
                "tree": DecisionTreeClassifier(max_depth=5, class_weight=cw, random_state=seed),
                "rf": RandomForestClassifier(300, class_weight=cw, random_state=seed, n_jobs=-1)}
            for name, m in models.items():
                pipe = build(m, num, FEATURES_CAT).fit(train[num + FEATURES_CAT], train[TARGET])
                X = test[num + FEATURES_CAT]
                res = metrics(test[TARGET], pipe.predict(X), pipe.predict_proba(X)[:, 1])
                rows.append({"condition": "manual", "model": name, "class_weight": cw or "none",
                             "leakage_included": leak, **res})
    out = pd.DataFrame(rows)
    out.to_csv(f"{outdir}/manual_metrics.csv", index=False)
    print(out[["model", "class_weight", "leakage_included", "accuracy", "recall", "f1", "roc_auc"]]
          .round(3).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/invoices.csv")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    main(a.data, a.test_frac, a.seed, a.outdir)
