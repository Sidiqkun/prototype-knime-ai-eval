"""Score predictions exported from a KNIME workflow (manual OR AI-built) with the SAME metric code.
In KNIME: use 'CSV Writer' after the Predictor node. Required columns: true + predicted label; optional: score.
Usage: python src/evaluate_predictions.py --pred knime_out.csv --run_id AI_mid_r1 \
        --true_col late_30 --pred_col "Prediction (late_30)" --score_col "P (late_30=1)"
"""
import argparse, json, os
import pandas as pd
from common import metrics

def main(a):
    df = pd.read_csv(a.pred)
    yt, yp = df[a.true_col].astype(int), df[a.pred_col].astype(int)
    ys = df[a.score_col] if a.score_col else None
    res = {"run_id": a.run_id, **metrics(yt, yp, ys)}
    print(json.dumps(res, indent=2))
    path = os.path.join(a.outdir, "knime_run_metrics.csv")
    pd.DataFrame([res]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--true_col", default="y_true")
    ap.add_argument("--pred_col", default="y_pred")
    ap.add_argument("--score_col", default=None)
    ap.add_argument("--outdir", default="results")
    main(ap.parse_args())
