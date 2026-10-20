"""Heuristic audit of an exported KNIME workflow (.knwf) for leakage columns and risky node settings.
It greps the XML settings inside the archive. It is a FIRST-PASS screen; confirm findings manually in KNIME.
Usage: python src/workflow_audit.py workflows/ai_run1.knwf [--out results/audit.csv]
"""
import argparse, os, re, zipfile
import pandas as pd
from common import LEAKAGE_COLS

RISKY = {"random_partition": re.compile(r"Draw randomly|Random", re.I),
         "smote_node": re.compile(r"SMOTE", re.I),
         "llm_prompter": re.compile(r"Prompter|LLM", re.I)}

def audit(path):
    findings = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.endswith(("settings.xml", "workflow.knime")):
                continue
            txt = z.read(n).decode("utf-8", errors="ignore")
            node = n.split("/")[-2] if "/" in n else n
            for col in LEAKAGE_COLS:
                if col in txt:
                    findings.append({"check": "leakage_column_referenced", "detail": col, "node": node})
            for name, rx in RISKY.items():
                if rx.search(txt):
                    findings.append({"check": name, "detail": rx.pattern, "node": node})
    return findings

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("knwf")
    ap.add_argument("--out", default="results/audit.csv")
    a = ap.parse_args()
    f = audit(a.knwf)
    df = pd.DataFrame(f).assign(workflow=os.path.basename(a.knwf)) if f else pd.DataFrame()
    print(df.to_string(index=False) if len(df) else "No findings (still verify manually).")
    if len(df):
        df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
