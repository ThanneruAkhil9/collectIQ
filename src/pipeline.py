"""
CollectIQ Pipeline Orchestrator
================================
The main pipeline that loads all models and runs the full daily routine:

  1. Identify overdue invoices
  2. Get each customer's segment (cached from nightly segmentation)
  3. Score each invoice with the late-payment XGBoost model
  4. Ask the action planner agent for a recommendation
  5. Draft the email
  6. Run outbound guardrails
  7. Add to action queue
  8. Project DSO impact

Returns a structured action queue that the Gradio UI displays.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from .segmentation.customer_segmenter import CustomerSegmenter
from .prediction.late_payment_model import LatePaymentPredictor
from .prediction.feature_engineering import build_invoice_features, FEATURE_COLUMNS
from .agent.llm_client import LLMClient
from .agent.action_planner import ActionPlanner
from .agent.tools import ActionDecision
from .drafting.email_drafter import EmailDrafter, DraftedEmail
from .drafting.tone_personas import get_tone_for_action
from .guardrails.outbound_validator import OutboundValidator, ValidationReport
from .simulation.dso_simulator import DSOSimulator, DSOReport
from .observability.langfuse_client import with_trace

load_dotenv()


@dataclass
class QueueItem:
    """A single recommended action with everything the user needs to act."""
    invoice: dict
    customer: dict
    days_overdue: int
    segment: str
    late_probability: float
    decision: ActionDecision
    email: Optional[DraftedEmail] = None
    validation: Optional[ValidationReport] = None

    def to_dict(self) -> dict:
        return {
            "invoice_id": self.invoice["invoice_id"],
            "customer_name": self.customer["name"],
            "customer_id": self.customer["customer_id"],
            "amount": float(self.invoice["total_amount"]),
            "days_overdue": self.days_overdue,
            "segment": self.segment,
            "late_probability": self.late_probability,
            "action": self.decision.tool_name,
            "reasoning": self.decision.reasoning,
            "arguments": self.decision.arguments,
            "email": self.email.to_dict() if self.email else None,
            "validation": self.validation.to_dict() if self.validation else None,
        }


@dataclass
class PipelineResult:
    queue: list[QueueItem]
    dso_report: DSOReport
    as_of: date

    def summary(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "num_actions": len(self.queue),
            "actions_by_type": self._count_actions(),
            "dso": self.dso_report.to_dict(),
        }

    def _count_actions(self) -> dict:
        from collections import Counter
        return dict(Counter(item.decision.tool_name for item in self.queue))


class CollectIQPipeline:
    """The end-to-end orchestrator."""

    def __init__(self, data_dir: Optional[Path] = None, draft_emails: bool = True):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1] / "data"
        self.data_dir = data_dir
        self.draft_emails = draft_emails

        logger.info(f"Loading data from {data_dir}...")
        self.customers = json.load(open(data_dir / "customers.json"))
        self.invoices = json.load(open(data_dir / "invoices.json"))
        self.payments = json.load(open(data_dir / "payments.json"))
        self.interactions = json.load(open(data_dir / "interactions.json"))
        try:
            self.customer_personas = json.load(open(data_dir / "customer_personas.json"))
        except FileNotFoundError:
            logger.warning("No segmentation found — using ground-truth personas")
            self.customer_personas = {c["customer_id"]: c["persona_truth"] for c in self.customers}

        self._customer_lookup = {c["customer_id"]: c for c in self.customers}

        # Models
        models_dir = data_dir / "models"
        try:
            self.predictor = LatePaymentPredictor.load(models_dir / "late_payment.xgb")
            logger.info("✅ Loaded late-payment classifier")
        except FileNotFoundError:
            logger.warning("⚠️  No trained late-payment model; predictions will be 0.5")
            self.predictor = None

        try:
            self.segmenter = CustomerSegmenter.load(models_dir / "segments.pkl")
            logger.info("✅ Loaded customer segmenter")
        except FileNotFoundError:
            logger.warning("⚠️  No segmenter; using cached customer_personas.json")
            self.segmenter = None

        # LLM-dependent components — only if API key is configured
        self.llm: Optional[LLMClient] = None
        self.planner: Optional[ActionPlanner] = None
        self.drafter: Optional[EmailDrafter] = None
        if os.getenv("GROQ_API_KEY"):
            self.llm = LLMClient()
            self.planner = ActionPlanner(self.llm)
            self.drafter = EmailDrafter(self.llm)
            logger.info("✅ LLM components initialized")
        else:
            logger.warning("⚠️  GROQ_API_KEY not set — pipeline runs in dry mode")

        self.validator = OutboundValidator()
        self.dso_sim = DSOSimulator(self.customers, self.invoices, self.payments,
                                       self.customer_personas)

    def get_overdue_invoices(self, as_of: Optional[date] = None,
                              min_days_overdue: int = 1) -> list[dict]:
        """Return all currently overdue/open invoices."""
        as_of = as_of or date.today()
        out = []
        for inv in self.invoices:
            if inv["status"] not in ("open", "overdue"):
                continue
            due = date.fromisoformat(inv["due_date"][:10])
            days_over = (as_of - due).days
            if days_over >= min_days_overdue:
                out.append(inv)
        return out

    def process_invoice(self, invoice: dict, as_of: date,
                         draft_email: bool = True, trace=None) -> QueueItem:
        """Run the full per-invoice pipeline."""
        customer = self._customer_lookup[invoice["customer_id"]]

        # 1. Features as of today
        features = build_invoice_features(
            invoice=invoice, customer=customer, as_of=as_of,
            paid_invoices=self.invoices, interactions=self.interactions,
            all_invoices=self.invoices,
        )
        days_overdue = features["days_overdue"]

        # 2. Segment
        segment = self.customer_personas.get(customer["customer_id"], "Standard")

        # 3. Late-payment probability
        if self.predictor:
            X = pd.DataFrame([features])[FEATURE_COLUMNS]
            prob = float(self.predictor.predict_proba(X)[0])
        else:
            prob = 0.5

        # 4. Agent decision
        if self.planner:
            decision = self.planner.decide(invoice, customer, features, segment, prob, trace=trace)
        else:
            # Dry-run rule-based fallback
            decision = self._dry_run_decision(invoice, segment, days_overdue, prob)

        # 5. Email draft (skip for 'wait' action)
        email = None
        validation = None
        if draft_email and decision.tool_name != "wait" and self.drafter:
            try:
                email = self.drafter.draft(invoice, customer, features,
                                            action=decision.tool_name,
                                            segment=segment, trace=trace)
                validation = self.validator.validate(email.subject, email.body,
                                                       invoice, customer)
            except Exception as e:
                logger.warning(f"Email drafting failed for {invoice['invoice_id']}: {e}")

        return QueueItem(
            invoice=invoice, customer=customer, days_overdue=days_overdue,
            segment=segment, late_probability=prob,
            decision=decision, email=email, validation=validation,
        )

    def _dry_run_decision(self, invoice: dict, segment: str,
                            days_overdue: int, prob: float) -> ActionDecision:
        """Rule-based fallback when LLM isn't available — for testing."""
        # Simple rules mirror what we'd expect the LLM to do
        if segment == "Strategic":
            tool = "send_friendly_reminder"
        elif segment == "Bad Debt" and days_overdue > 90:
            tool = "escalate_to_legal"
        elif segment == "At Risk":
            tool = "propose_payment_plan"
        elif segment == "Chronic Late" and days_overdue > 21:
            tool = "send_firm_escalation"
        elif days_overdue > 60:
            tool = "flag_for_credit_hold"
        elif days_overdue > 15:
            tool = "send_firm_escalation"
        else:
            tool = "send_friendly_reminder"
        return ActionDecision(
            invoice_id=invoice["invoice_id"], tool_name=tool,
            arguments={"cc_manager": days_overdue > 30} if "escalation" in tool else {},
            reasoning=f"[Dry-run rule] {segment} customer, {days_overdue} days overdue, "
                       f"late probability {prob:.0%}.",
            confidence=prob, customer_segment=segment,
        )

    def run(self, as_of: Optional[date] = None,
            max_invoices: Optional[int] = None,
            draft_emails: Optional[bool] = None,
            trace_name: str = "collectiq_daily_run") -> PipelineResult:
        """Run the full pipeline. Returns a queue + DSO projection."""
        as_of = as_of or date.today()
        if draft_emails is None:
            draft_emails = self.draft_emails

        overdue = self.get_overdue_invoices(as_of=as_of)
        # Sort by most overdue first
        overdue.sort(key=lambda inv: (as_of - date.fromisoformat(inv["due_date"][:10])).days,
                      reverse=True)
        if max_invoices:
            overdue = overdue[:max_invoices]
        logger.info(f"Processing {len(overdue)} overdue invoices...")

        queue: list[QueueItem] = []
        with with_trace(trace_name, user_id="ar_team") as trace:
            for inv in overdue:
                try:
                    item = self.process_invoice(inv, as_of, draft_email=draft_emails,
                                                  trace=trace)
                    queue.append(item)
                except Exception as e:
                    logger.warning(f"Failed to process {inv['invoice_id']}: {e}")

        # Project DSO impact
        dso_report = self.dso_sim.project_with_actions([item.decision for item in queue])

        return PipelineResult(queue=queue, dso_report=dso_report, as_of=as_of)


# ---- Smoke test ----
if __name__ == "__main__":
    pipe = CollectIQPipeline(draft_emails=False)
    result = pipe.run(max_invoices=10)
    print(f"\n=== CollectIQ Daily Run — {result.as_of} ===")
    print(f"Processed {len(result.queue)} overdue invoices")
    print(f"\n{result.dso_report.summary()}")
    print(f"\nFirst 5 recommendations:")
    for item in result.queue[:5]:
        print(f"  {item.invoice['invoice_id']:18} "
                f"({item.customer['name'][:30]:30}) "
                f"{item.days_overdue:3}d overdue · "
                f"{item.segment:13} · "
                f"P(late)={item.late_probability:.0%} · "
                f"→ {item.decision.tool_name}")
