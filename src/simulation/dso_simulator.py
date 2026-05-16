"""
DSO Simulator
=============
Projects the impact of CollectIQ's recommendations on:
  - Days Sales Outstanding (DSO)
  - Working capital tied up in receivables
  - Per-action expected reduction in payment delay

This is the metric that justifies the project to a CFO. Translates
ML output into rupees.

DSO formula:
  DSO = (Total AR Balance / Total Revenue) × Number of Days

Simulation logic:
  Each action has an empirical "expected delay reduction" — e.g., a firm
  escalation to a Chronic Late customer typically shaves ~12 days off
  the eventual payment. These coefficients come from industry studies +
  some calibration against the synthetic data.
"""
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict


# Empirical delay reductions per (action, segment) combination
# These are conservative; calibrated against synthetic data behavior
DELAY_REDUCTION_DAYS = {
    # action_name: {segment: avg_days_reduction}
    "send_friendly_reminder": {
        "Strategic": 3, "Standard": 4, "Chronic Late": 1,
        "At Risk": 0, "Bad Debt": 0,
    },
    "send_firm_escalation": {
        "Strategic": 5, "Standard": 8, "Chronic Late": 12,
        "At Risk": 6, "Bad Debt": 2,
    },
    "propose_payment_plan": {
        # Payment plan converts a write-off into a recovery; treat as +25 days off
        "Strategic": 10, "Standard": 15, "Chronic Late": 18,
        "At Risk": 25, "Bad Debt": 30,
    },
    "flag_for_credit_hold": {
        # Credit hold forces a conversation — strong reduction on chronic offenders
        "Strategic": 0, "Standard": 10, "Chronic Late": 20,
        "At Risk": 15, "Bad Debt": 8,
    },
    "escalate_to_legal": {
        # Legal escalation either gets payment or moves to write-off
        "Strategic": 0, "Standard": 5, "Chronic Late": 15,
        "At Risk": 20, "Bad Debt": 25,
    },
    "wait": {
        # Wait = take no action; assumed equal to baseline
        "Strategic": 0, "Standard": 0, "Chronic Late": 0,
        "At Risk": 0, "Bad Debt": 0,
    },
}


@dataclass
class DSOReport:
    baseline_dso_days: float
    projected_dso_days: float
    dso_reduction_days: float
    baseline_ar_balance: float
    projected_ar_balance: float
    working_capital_unlocked: float
    annual_revenue: float
    num_actions: int
    actions_by_type: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"DSO: {self.baseline_dso_days:.1f} → {self.projected_dso_days:.1f} days "
            f"(−{self.dso_reduction_days:.1f} days)\n"
            f"Working capital unlocked: ₹{self.working_capital_unlocked:,.0f}\n"
            f"Based on {self.num_actions} recommended actions"
        )

    def to_dict(self) -> dict:
        return {
            "baseline_dso_days": self.baseline_dso_days,
            "projected_dso_days": self.projected_dso_days,
            "dso_reduction_days": self.dso_reduction_days,
            "baseline_ar_balance": self.baseline_ar_balance,
            "projected_ar_balance": self.projected_ar_balance,
            "working_capital_unlocked": self.working_capital_unlocked,
            "annual_revenue": self.annual_revenue,
            "num_actions": self.num_actions,
            "actions_by_type": self.actions_by_type,
        }


class DSOSimulator:
    """Computes baseline DSO + projects impact of agent actions."""

    def __init__(self, customers: list[dict], invoices: list[dict],
                 payments: list[dict], customer_personas: dict[str, str],
                 as_of: Optional[date] = None):
        self.customers = {c["customer_id"]: c for c in customers}
        self.invoices = invoices
        self.payments = payments
        self.customer_personas = customer_personas
        self.as_of = as_of or date.today()

    def compute_baseline_dso(self, lookback_days: int = 365) -> tuple[float, float, float]:
        """
        Compute current DSO using the standard formula:
          DSO = (AR balance / revenue in lookback window) × lookback_days

        Returns (dso_days, ar_balance, annual_revenue_proxy)
        """
        cutoff = self.as_of - timedelta(days=lookback_days)
        ar_balance = 0.0
        revenue = 0.0
        for inv in self.invoices:
            issue = date.fromisoformat(inv["issue_date"][:10])
            if issue < cutoff:
                continue
            total = float(inv["total_amount"])
            revenue += total
            if inv["status"] in ("open", "overdue"):
                ar_balance += total

        if revenue == 0:
            return 0.0, ar_balance, 0.0

        dso = (ar_balance / revenue) * lookback_days
        # Approximate annual revenue from lookback
        annual_revenue = revenue * (365.0 / lookback_days)
        return dso, ar_balance, annual_revenue

    def project_with_actions(self, decisions: list, lookback_days: int = 365) -> DSOReport:
        """
        Given a list of ActionDecision objects, project the new DSO assuming
        all actions get their expected delay reduction.

        We simulate this by reducing the "effective overdue days" of each
        affected invoice, which proportionally reduces the AR balance.
        """
        baseline_dso, baseline_ar, annual_revenue = self.compute_baseline_dso(lookback_days)

        # Compute total "saved days × amount" from all actions
        total_dso_reduction_basis = 0.0  # in (rupees × days)
        actions_by_type: dict[str, int] = defaultdict(int)

        invoice_lookup = {inv["invoice_id"]: inv for inv in self.invoices}

        for d in decisions:
            invoice = invoice_lookup.get(d.invoice_id)
            if not invoice:
                continue
            segment = d.customer_segment or "Standard"
            reduction = DELAY_REDUCTION_DAYS.get(d.tool_name, {}).get(segment, 0)
            amount = float(invoice["total_amount"])
            total_dso_reduction_basis += reduction * amount
            actions_by_type[d.tool_name] += 1

        # Conceptually: reducing N days of delay on ₹X means ₹X stays in AR for N fewer days
        # AR_reduction_per_lookback_window = total_dso_reduction_basis / lookback_days
        ar_reduction = total_dso_reduction_basis / lookback_days
        projected_ar = max(0, baseline_ar - ar_reduction)

        # New DSO
        if annual_revenue > 0:
            projected_dso = (projected_ar / annual_revenue) * 365.0
        else:
            projected_dso = baseline_dso

        return DSOReport(
            baseline_dso_days=round(baseline_dso, 1),
            projected_dso_days=round(projected_dso, 1),
            dso_reduction_days=round(baseline_dso - projected_dso, 1),
            baseline_ar_balance=round(baseline_ar, 2),
            projected_ar_balance=round(projected_ar, 2),
            working_capital_unlocked=round(baseline_ar - projected_ar, 2),
            annual_revenue=round(annual_revenue, 2),
            num_actions=len(decisions),
            actions_by_type=dict(actions_by_type),
        )


# ---- Smoke test ----
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    customers = json.load(open(data_dir / "customers.json"))
    invoices = json.load(open(data_dir / "invoices.json"))
    payments = json.load(open(data_dir / "payments.json"))
    try:
        personas = json.load(open(data_dir / "customer_personas.json"))
    except FileNotFoundError:
        personas = {c["customer_id"]: c["persona_truth"] for c in customers}

    sim = DSOSimulator(customers, invoices, payments, personas)
    dso, ar, revenue = sim.compute_baseline_dso()
    print(f"Baseline DSO: {dso:.1f} days")
    print(f"AR Balance: ₹{ar:,.0f}")
    print(f"Annual Revenue (estimated): ₹{revenue:,.0f}")

    # Project with empty actions list — should give baseline
    report = sim.project_with_actions([])
    print(f"\n{report.summary()}")
