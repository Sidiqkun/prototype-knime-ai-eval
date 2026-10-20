"""Generate synthetic invoice data (PT Maju Bersama style, 2024-07-01 .. 2026-06-30).
Target: late_30 = 1 if invoice unpaid/paid > 30 days after due date.
Leakage columns (NOT available at invoice issue time): paid_date, days_overdue, final_status.
Usage: python src/generate_data.py --seed 42
"""
import argparse
import numpy as np
import pandas as pd

def main(seed: int, n_inv: int, n_cust: int, out: str):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2024-07-01"), pd.Timestamp("2026-06-30")
    cust = pd.DataFrame({
        "customer_id": np.arange(1, n_cust + 1),
        "customer_class": rng.choice(["A", "B", "C"], n_cust, p=[.25, .45, .30]),
        "risk": rng.beta(1.2, 9, n_cust),  # hidden latent risk
    })
    days = (end - start).days
    inv = pd.DataFrame({
        "invoice_id": np.arange(1, n_inv + 1),
        "customer_id": rng.integers(1, n_cust + 1, n_inv),
        "invoice_date": start + pd.to_timedelta(rng.integers(0, days, n_inv), unit="D"),
        "payment_method": rng.choice(["transfer", "cash", "credit", "giro"], n_inv, p=[.4, .15, .35, .10]),
        "credit_terms": rng.choice([0, 14, 30, 45], n_inv, p=[.15, .25, .4, .2]),
        "n_items": rng.integers(1, 8, n_inv),
        "discount_pct": np.round(rng.choice([0, 0, 0, 2, 5, 10, 15], n_inv), 1),
    })
    inv["invoice_value"] = np.round(rng.lognormal(15, .6, n_inv), -3)
    inv = inv.merge(cust, on="customer_id").sort_values("invoice_date").reset_index(drop=True)

    z = (-3.2 + 6 * inv["risk"] + 0.9 * (inv["payment_method"] == "credit")
         + 0.7 * (inv["payment_method"] == "giro") + 0.012 * inv["credit_terms"]
         + 0.35 * (np.log(inv["invoice_value"]) - 15) + 0.04 * inv["discount_pct"])
    p_late = 1 / (1 + np.exp(-z))
    inv["late_30"] = (rng.random(n_inv) < p_late).astype(int)

    inv["due_date"] = inv["invoice_date"] + pd.to_timedelta(inv["credit_terms"], unit="D")
    extra = np.where(inv["late_30"] == 1, rng.integers(31, 120, n_inv), rng.integers(-10, 25, n_inv))
    inv["paid_date"] = inv["due_date"] + pd.to_timedelta(extra, unit="D")
    unpaid = (inv["late_30"] == 1) & (rng.random(n_inv) < .3)
    inv.loc[unpaid, "paid_date"] = pd.NaT
    inv["days_overdue"] = np.where(unpaid, (end - inv["due_date"]).dt.days.clip(lower=31), extra)
    inv["final_status"] = np.where(unpaid, "UNPAID", np.where(inv["late_30"] == 1, "PAID_LATE", "PAID_OK"))

    # leak-free customer history: share of late outcomes KNOWN at invoice date (due+30 already passed)
    hist = np.full(n_inv, np.nan); cnt = np.zeros(n_inv)
    for _, g in inv.groupby("customer_id"):
        idx = g.index.to_numpy()
        known_at = (g["due_date"] + pd.Timedelta(days=30)).to_numpy()
        dates = g["invoice_date"].to_numpy(); y = g["late_30"].to_numpy()
        for k, i in enumerate(idx):
            m = known_at < dates[k]
            cnt[i] = m.sum(); hist[i] = y[m].mean() if m.any() else np.nan
    inv["prior_invoice_count"], inv["prior_late_rate"] = cnt, hist
    inv.drop(columns=["risk"]).to_csv(out, index=False)  # hidden latent risk is not exported
    print(f"saved {out}: {len(inv)} rows, late rate = {inv['late_30'].mean():.1%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_invoices", type=int, default=4000)
    ap.add_argument("--n_customers", type=int, default=120)
    ap.add_argument("--out", default="data/invoices.csv")
    a = ap.parse_args()
    main(a.seed, a.n_invoices, a.n_customers, a.out)
