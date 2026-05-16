"""
Customer Segmenter
==================
Segments customers into 5 personas based on payment behavior using k-means
on 8 derived features. Outputs a label per customer that the agent uses
to choose tone and aggressiveness.

Features used:
  - avg_days_late: mean payment delay vs due date
  - std_days_late: variability in delay
  - pct_paid_late: fraction of paid invoices that were >5 days late
  - pct_severe_late: fraction >30 days late
  - num_invoices_24mo: total volume (proxy for relationship size)
  - avg_invoice_amount: relationship value
  - days_since_last_payment: freshness
  - delay_trend: slope of delay over last 12 months (deterioration signal)

The k-means output is then mapped to human-readable labels:
  Strategic, Standard, Chronic Late, At Risk, Bad Debt

This is intentionally interpretable — finance teams need to defend why a
customer is in a particular bucket.
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib


PERSONA_NAMES = ["Strategic", "Standard", "Chronic Late", "At Risk", "Bad Debt"]


def _to_date(s):
    if not s:
        return None
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s)[:10])


def build_customer_features(customers: list[dict],
                             invoices: list[dict],
                             payments: list[dict],
                             as_of: Optional[date] = None) -> pd.DataFrame:
    """
    Build the feature matrix used for segmentation.
    Returns a DataFrame indexed by customer_id.
    """
    if as_of is None:
        as_of = date.today()

    # Index data
    pays_by_cust = defaultdict(list)
    for p in payments:
        pays_by_cust[p["customer_id"]].append(p)
    invs_by_cust = defaultdict(list)
    for inv in invoices:
        invs_by_cust[inv["customer_id"]].append(inv)

    rows = []
    for c in customers:
        cust_id = c["customer_id"]
        cust_pays = pays_by_cust[cust_id]
        cust_invs = invs_by_cust[cust_id]

        if cust_pays:
            delays = [p["days_late"] for p in cust_pays]
            avg_delay = float(np.mean(delays))
            std_delay = float(np.std(delays))
            pct_late = sum(1 for d in delays if d > 5) / len(delays)
            pct_severe = sum(1 for d in delays if d > 30) / len(delays)
            last_pay = max(_to_date(p["payment_date"]) for p in cust_pays)
            days_since_last = (as_of - last_pay).days
        else:
            avg_delay = 0.0
            std_delay = 0.0
            pct_late = 0.0
            pct_severe = 0.0
            days_since_last = 999

        # Delay trend: compare delay in last 6 months vs delay 7-12 months ago
        recent_cutoff = as_of - timedelta(days=180)
        older_cutoff = as_of - timedelta(days=360)
        recent_delays = [p["days_late"] for p in cust_pays
                          if _to_date(p["payment_date"]) >= recent_cutoff]
        older_delays = [p["days_late"] for p in cust_pays
                         if older_cutoff <= _to_date(p["payment_date"]) < recent_cutoff]
        if recent_delays and older_delays:
            delay_trend = float(np.mean(recent_delays) - np.mean(older_delays))
        else:
            delay_trend = 0.0

        avg_amount = float(np.mean([inv["total_amount"] for inv in cust_invs])) if cust_invs else 0.0

        rows.append({
            "customer_id": cust_id,
            "avg_days_late": avg_delay,
            "std_days_late": std_delay,
            "pct_paid_late": pct_late,
            "pct_severe_late": pct_severe,
            "num_invoices_24mo": len(cust_invs),
            "avg_invoice_amount": avg_amount,
            "days_since_last_payment": days_since_last,
            "delay_trend": delay_trend,
        })

    df = pd.DataFrame(rows).set_index("customer_id")
    return df


class CustomerSegmenter:
    """
    K-means based customer segmentation.

    Trains on 8 features → 5 clusters → mapped to interpretable persona names
    by ranking clusters on a 'risk score' computed from avg_delay + pct_severe.
    """

    FEATURES = [
        "avg_days_late", "std_days_late", "pct_paid_late", "pct_severe_late",
        "num_invoices_24mo", "avg_invoice_amount", "days_since_last_payment",
        "delay_trend",
    ]

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_to_persona: dict[int, str] = {}

    def fit(self, feature_df: pd.DataFrame) -> "CustomerSegmenter":
        """Fit the segmenter."""
        X = feature_df[self.FEATURES].values
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        # Assign labels temporarily to compute mapping
        labels = self.kmeans.labels_
        # Compute cluster-level risk score
        risk_scores = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if mask.sum() == 0:
                risk_scores[cluster_id] = 0.0
                continue
            cluster_data = feature_df.iloc[mask]
            # Risk = weighted combination of delay metrics + amount (inverse — strategic has higher amount)
            risk = (
                cluster_data["avg_days_late"].mean() * 1.0 +
                cluster_data["pct_severe_late"].mean() * 100.0 +
                cluster_data["delay_trend"].mean() * 2.0 +
                cluster_data["days_since_last_payment"].mean() * 0.3
            )
            risk_scores[cluster_id] = float(risk)
        # Sort clusters by risk ascending → map to PERSONA_NAMES
        sorted_clusters = sorted(risk_scores.keys(), key=lambda k: risk_scores[k])
        # Strategic has lowest avg_days_late but HIGH avg_invoice_amount
        # Need to differentiate Strategic from Standard
        # Take the two lowest-risk clusters and pick the higher-amount one as Strategic
        lowest = sorted_clusters[:2]
        if len(lowest) == 2:
            amounts = {c: feature_df.iloc[labels == c]["avg_invoice_amount"].mean()
                        for c in lowest}
            strategic_cluster = max(amounts, key=amounts.get)
            standard_cluster = min(amounts, key=amounts.get)
            ordered = [strategic_cluster, standard_cluster] + sorted_clusters[2:]
        else:
            ordered = sorted_clusters
        # Map to PERSONA_NAMES (truncated/padded if needed)
        self.cluster_to_persona = {
            cluster_id: PERSONA_NAMES[i] if i < len(PERSONA_NAMES) else f"Cluster {i}"
            for i, cluster_id in enumerate(ordered)
        }
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """Predict persona labels."""
        X = feature_df[self.FEATURES].values
        X_scaled = self.scaler.transform(X)
        labels = self.kmeans.predict(X_scaled)
        personas = pd.Series(
            [self.cluster_to_persona.get(l, "Unknown") for l in labels],
            index=feature_df.index,
            name="persona",
        )
        return personas

    def save(self, path: Path):
        joblib.dump({
            "scaler": self.scaler, "kmeans": self.kmeans,
            "cluster_to_persona": self.cluster_to_persona,
            "n_clusters": self.n_clusters,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "CustomerSegmenter":
        data = joblib.load(path)
        obj = cls(n_clusters=data["n_clusters"])
        obj.scaler = data["scaler"]
        obj.kmeans = data["kmeans"]
        obj.cluster_to_persona = data["cluster_to_persona"]
        return obj


def main():
    """Run segmentation end-to-end and print summary."""
    data_dir = Path(__file__).resolve().parents[2] / "data"
    customers = json.load(open(data_dir / "customers.json"))
    invoices = json.load(open(data_dir / "invoices.json"))
    payments = json.load(open(data_dir / "payments.json"))

    features = build_customer_features(customers, invoices, payments)
    print(f"Built features for {len(features)} customers")
    print(features.head().round(2))

    segmenter = CustomerSegmenter(n_clusters=5).fit(features)
    personas = segmenter.predict(features)

    # Save model + per-customer assignment
    models_dir = data_dir / "models"
    models_dir.mkdir(exist_ok=True)
    segmenter.save(models_dir / "segments.pkl")

    # Save customer → persona mapping
    customer_personas = {cid: persona for cid, persona in personas.items()}
    with open(data_dir / "customer_personas.json", "w") as f:
        json.dump(customer_personas, f, indent=2)

    # Compare to ground truth
    truth = {c["customer_id"]: c["persona_truth"] for c in customers}
    matches = sum(1 for cid, p in personas.items() if p == truth.get(cid))
    print(f"\n✅ Segmentation complete. Cluster→persona mapping:")
    for cluster_id, persona in segmenter.cluster_to_persona.items():
        count = (segmenter.kmeans.labels_ == cluster_id).sum()
        print(f"   Cluster {cluster_id} → {persona:13} ({count} customers)")
    print(f"\n   Accuracy vs ground truth: {matches}/{len(personas)} ({matches/len(personas)*100:.1f}%)")
    print(f"   Model saved → {models_dir / 'segments.pkl'}")


if __name__ == "__main__":
    main()
