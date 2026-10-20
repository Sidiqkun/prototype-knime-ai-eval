"""Aggregate experiment logs: manual vs KNIME-AI, by prompt level, and error map per CRISP-DM stage.
Inputs : data/experiments.csv , data/errors.csv  (see *_template.csv)
Usage  : python src/analyze_results.py
"""
import pandas as pd
from scipy.stats import mannwhitneyu

def cliffs_delta(a, b):
    gt = sum(x > y for x in a for y in b); lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def main():
    ex = pd.read_csv("data/experiments.csv")
    er = pd.read_csv("data/errors.csv")
    cols = ["minutes", "n_errors", "human_corrections", "recall", "f1", "roc_auc"]
    print("== Summary by task x condition ==")
    s = ex.groupby(["task", "condition"])[cols].agg(["mean", "std"]).round(3)
    print(s.to_string()); s.to_csv("results/summary_by_condition.csv")

    print("\n== AI by prompt level ==")
    ai = ex[ex.condition == "ai"]
    p = ai.groupby("prompt_level")[cols].mean().round(3)
    print(p.to_string()); p.to_csv("results/ai_by_prompt_level.csv")

    print("\n== Manual vs AI tests (Mann-Whitney U, Cliff's delta) ==")
    rows = []
    for c in cols:
        m, a = ex[ex.condition == "manual"][c].dropna(), ex[ex.condition == "ai"][c].dropna()
        if len(m) > 1 and len(a) > 1:
            u, pv = mannwhitneyu(m, a, alternative="two-sided")
            rows.append({"metric": c, "U": u, "p_value": pv, "cliffs_delta": cliffs_delta(list(m), list(a))})
    t = pd.DataFrame(rows).round(4)
    print(t.to_string(index=False) if len(t) else "Not enough runs per condition (need >=2 each).")
    t.to_csv("results/tests.csv", index=False)

    print("\n== Error map: CRISP-DM stage x error type ==")
    em = pd.crosstab(er["crisp_stage"], er["error_type"], margins=True)
    print(em.to_string()); em.to_csv("results/error_map.csv")

    print("\n== Severity by stage ==")
    sv = er.groupby("crisp_stage")["severity"].agg(["count", "mean"]).round(2)
    print(sv.to_string()); sv.to_csv("results/severity_by_stage.csv")

if __name__ == "__main__":
    main()
