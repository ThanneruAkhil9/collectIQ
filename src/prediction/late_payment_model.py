"""
Late-Payment Predictor (XGBoost)
=================================
Binary classifier: given features for an open invoice + its customer's
history, predict P(invoice will be 30+ days late at payment).

Trained on historical paid invoices with features computed AT ISSUE TIME
(not at payment time — that would leak the answer).

Target metric: F1 on the positive class (30+ days late).
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import joblib

from .feature_engineering import FEATURE_COLUMNS, build_training_set, build_invoice_features


class LatePaymentPredictor:
    """XGBoost classifier for P(invoice 30+ days late)."""

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 random_state: int = 42):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": 4,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        self.model = xgb.XGBClassifier(**self.params)
        if X_val is not None and y_val is not None:
            self.model.fit(X[self.feature_columns], y,
                            eval_set=[(X_val[self.feature_columns], y_val)],
                            verbose=False)
        else:
            self.model.fit(X[self.feature_columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability of class 1 (late)."""
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(ascending=False)

    def save(self, path: Path):
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "params": self.params,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "LatePaymentPredictor":
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.feature_columns = data["feature_columns"]
        obj.params = data["params"]
        return obj


def train_and_evaluate(data_dir: Path = None, save_model: bool = True) -> dict:
    """Train on 80% of historical paid invoices, evaluate on 20% held-out."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"

    customers = json.load(open(data_dir / "customers.json"))
    invoices = json.load(open(data_dir / "invoices.json"))
    payments = json.load(open(data_dir / "payments.json"))
    interactions = json.load(open(data_dir / "interactions.json"))

    print(f"Building training set...")
    df = build_training_set(customers, invoices, payments, interactions)
    print(f"  Training set: {len(df)} invoices, {df['is_late'].mean():.1%} late")

    X = df[FEATURE_COLUMNS]
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print(f"\nTraining XGBoost...")
    predictor = LatePaymentPredictor()
    predictor.fit(X_train, y_train, X_test, y_test)

    # Evaluate
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)
    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }

    print(f"\n📊 Late-payment classifier (held-out test, 30+ day threshold):")
    print(f"   F1        : {metrics['f1']:.3f}")
    print(f"   AUC-ROC   : {metrics['auc']:.3f}")
    print(f"   Precision : {metrics['precision']:.3f}")
    print(f"   Recall    : {metrics['recall']:.3f}")
    print(f"\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"           pred:0   pred:1")
    print(f"   true:0   {cm[0,0]:>4}    {cm[0,1]:>4}")
    print(f"   true:1   {cm[1,0]:>4}    {cm[1,1]:>4}")

    print(f"\nTop 8 feature importances:")
    for feat, imp in predictor.feature_importance().head(8).items():
        print(f"   {feat:30}  {imp:.3f}")

    if save_model:
        models_dir = data_dir / "models"
        models_dir.mkdir(exist_ok=True)
        predictor.save(models_dir / "late_payment.xgb")
        print(f"\n💾 Saved model → {models_dir / 'late_payment.xgb'}")

        # Save eval metrics for README
         (data_dir / "eval").mkdir(parents=True, exist_ok=True)
        with open(data_dir / "eval" / "classifier_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return {"predictor": predictor, "metrics": metrics}


if __name__ == "__main__":
    train_and_evaluate()
