"""
Feature Engineering for Late-Payment Prediction
=================================================
Turns a (customer, invoice) pair into a feature vector for the XGBoost
classifier. Features capture:
  - Invoice-level: amount, age, time since issue
  - Customer-level history: avg delay, severity, trend
  - Behavioural: prior interactions, open balance
  - Calendar: month seasonality

Used both at training time (on historical paid invoices) and at inference
time (on currently open/overdue invoices).
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd


def _to_date(s):
    if not s:
        return None
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s)[:10])


FEATURE_COLUMNS = [
    "invoice_amount",
    "days_since_issue",
    "days_overdue",                # 0 if not yet overdue
    "credit_terms_days",
    "amount_vs_customer_avg",      # ratio: this invoice / customer's avg
    "customer_avg_days_late",
    "customer_std_days_late",
    "customer_pct_severe_late",
    "customer_open_balance",       # sum of all currently-open invoices for this customer
    "customer_num_open_invoices",
    "customer_delay_trend",        # last 6mo vs prior 6mo
    "days_since_last_payment",
    "num_prior_interactions",      # collection emails sent to this customer
    "num_firm_interactions",       # firm/final_notice tone count
    "month_of_year",               # 1-12 seasonality
    "is_msme_window",              # within MSME 45-day rule
]


def _customer_history_features(customer_id: str,
                                up_to: date,
                                paid_invoices: list[dict],
                                interactions: list[dict],
                                all_invoices: list[dict]) -> dict:
    """
    Compute customer-history features as of `up_to` (exclusive).
    """
    # Filter paid invoices to those PAID BEFORE up_to (avoid leakage)
    cust_paid = [
        p for p in paid_invoices
        if p["customer_id"] == customer_id and _to_date(p.get("paid_date")) and _to_date(p["paid_date"]) < up_to
    ]

    if cust_paid:
        delays = [p["days_late_at_payment"] for p in cust_paid]
        avg_delay = float(np.mean(delays))
        std_delay = float(np.std(delays))
        pct_severe = sum(1 for d in delays if d > 30) / len(delays)
        last_pay_date = max(_to_date(p["paid_date"]) for p in cust_paid)
        days_since_last = (up_to - last_pay_date).days
        # Trend
        recent_cutoff = up_to - timedelta(days=180)
        older_cutoff = up_to - timedelta(days=360)
        recent = [p["days_late_at_payment"] for p in cust_paid
                  if _to_date(p["paid_date"]) >= recent_cutoff]
        older = [p["days_late_at_payment"] for p in cust_paid
                 if older_cutoff <= _to_date(p["paid_date"]) < recent_cutoff]
        delay_trend = float(np.mean(recent) - np.mean(older)) if (recent and older) else 0.0
        cust_avg_amount = float(np.mean([p["total_amount"] for p in cust_paid]))
    else:
        avg_delay = 0.0
        std_delay = 0.0
        pct_severe = 0.0
        days_since_last = 999
        delay_trend = 0.0
        cust_avg_amount = 0.0

    # Open balance: invoices for this customer with status open/overdue AS OF up_to
    # Approximation: count invoices issued before up_to and not yet paid by up_to
    open_count = 0
    open_balance = 0.0
    for inv in all_invoices:
        if inv["customer_id"] != customer_id:
            continue
        issue = _to_date(inv["issue_date"])
        if issue is None or issue >= up_to:
            continue
        # Not paid yet at up_to?
        paid = _to_date(inv.get("paid_date"))
        if paid is None or paid >= up_to:
            open_count += 1
            open_balance += float(inv["total_amount"])

    # Interactions before up_to
    cust_ints = [
        i for i in interactions
        if i["customer_id"] == customer_id and _to_date(i["date"]) < up_to
    ]
    num_int = len(cust_ints)
    num_firm = sum(1 for i in cust_ints if i["tone"] in ("firm", "final_notice"))

    return {
        "customer_avg_days_late": avg_delay,
        "customer_std_days_late": std_delay,
        "customer_pct_severe_late": pct_severe,
        "customer_open_balance": open_balance,
        "customer_num_open_invoices": open_count,
        "customer_delay_trend": delay_trend,
        "days_since_last_payment": days_since_last,
        "num_prior_interactions": num_int,
        "num_firm_interactions": num_firm,
        "_cust_avg_amount": cust_avg_amount,   # helper, dropped before return
    }


def build_invoice_features(invoice: dict,
                            customer: dict,
                            as_of: date,
                            paid_invoices: list[dict],
                            interactions: list[dict],
                            all_invoices: list[dict]) -> dict:
    """
    Build a single feature row for one invoice as of `as_of` date.
    """
    issue_date = _to_date(invoice["issue_date"])
    due_date = _to_date(invoice["due_date"])
    amount = float(invoice["total_amount"])
    days_since_issue = (as_of - issue_date).days if issue_date else 0
    days_overdue = max(0, (as_of - due_date).days) if due_date else 0
    credit_terms = customer["credit_terms_days"]
    is_msme_window = 1 if days_since_issue <= 45 else 0
    month_of_year = issue_date.month if issue_date else 1

    hist = _customer_history_features(
        customer["customer_id"], as_of, paid_invoices, interactions, all_invoices
    )
    cust_avg = hist.pop("_cust_avg_amount")
    amount_vs_avg = amount / cust_avg if cust_avg > 0 else 1.0

    return {
        "invoice_amount": amount,
        "days_since_issue": days_since_issue,
        "days_overdue": days_overdue,
        "credit_terms_days": credit_terms,
        "amount_vs_customer_avg": amount_vs_avg,
        "month_of_year": month_of_year,
        "is_msme_window": is_msme_window,
        **hist,
    }


def build_training_set(customers: list[dict],
                        invoices: list[dict],
                        payments: list[dict],
                        interactions: list[dict],
                        target_threshold: int = 30) -> pd.DataFrame:
    """
    Build a labeled training set from PAID invoices only.
    Target: is the invoice 30+ days late (=1) or not (=0)?

    To avoid leakage, each invoice's features are computed using only
    customer history AS OF the issue_date (not the payment date).
    """
    cust_lookup = {c["customer_id"]: c for c in customers}

    rows = []
    for inv in invoices:
        if inv["status"] != "paid" or inv.get("days_late_at_payment") is None:
            continue
        customer = cust_lookup.get(inv["customer_id"])
        if customer is None:
            continue
        issue_date = _to_date(inv["issue_date"])
        if issue_date is None:
            continue

        features = build_invoice_features(
            invoice=inv,
            customer=customer,
            as_of=issue_date,         # CRITICAL: features computed at issue time only
            paid_invoices=invoices,    # only those paid before issue_date will be used
            interactions=interactions,
            all_invoices=invoices,
        )
        # Target = was this invoice eventually 30+ days late?
        features["is_late"] = 1 if inv["days_late_at_payment"] >= target_threshold else 0
        features["invoice_id"] = inv["invoice_id"]
        features["customer_id"] = inv["customer_id"]
        rows.append(features)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    customers = json.load(open(data_dir / "customers.json"))
    invoices = json.load(open(data_dir / "invoices.json"))
    payments = json.load(open(data_dir / "payments.json"))
    interactions = json.load(open(data_dir / "interactions.json"))

    df = build_training_set(customers, invoices, payments, interactions)
    print(f"Built training set: {df.shape}")
    print(f"Late rate (30+ days): {df['is_late'].mean():.2%}")
    print(df[FEATURE_COLUMNS + ['is_late']].head().round(2))
