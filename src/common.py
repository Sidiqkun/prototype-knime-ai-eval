from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

LEAKAGE_COLS = ["paid_date", "days_overdue", "final_status"]
FEATURES_NUM = ["credit_terms", "n_items", "discount_pct", "invoice_value",
                "prior_invoice_count", "prior_late_rate"]
FEATURES_CAT = ["payment_method", "customer_class"]
TARGET = "late_30"

def metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1": f1_score(y_true, y_pred, zero_division=0),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if y_score is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out
