"""
Action Planner Agent
====================
The LLM brain. For each overdue invoice, takes:
  - Invoice details
  - Customer segment + payment history summary
  - Late-payment probability from the XGBoost model

And picks ONE tool from the 6 available actions, with reasoning.

Uses Groq's function-calling interface (Llama 3.3 70B supports tools).
Falls back to JSON parsing if function calling is unavailable.
"""
import json
import re
from typing import Optional
from loguru import logger

from .llm_client import LLMClient
from .tools import ACTION_TOOLS, ActionDecision


SYSTEM_PROMPT = """You are CollectIQ, an autonomous AR collections agent for a mid-sized Indian company.

For each overdue invoice you receive, you must pick exactly ONE action from the tools provided. Your goal is to maximize cash collected while preserving customer relationships.

DECISION FRAMEWORK:
- For 'Strategic' customers (high revenue, reliable payers): default to send_friendly_reminder. Avoid escalation unless severely overdue (>60 days).
- For 'Standard' customers: friendly reminder for 1-15 days overdue; firm escalation for 21-45 days overdue.
- For 'Chronic Late' customers: skip friendly reminders (they don't work); go directly to firm escalation with CC to manager.
- For 'At Risk' customers (deteriorating trend, multiple missed promises): propose_payment_plan to recover some cash, or flag_for_credit_hold.
- For 'Bad Debt' customers: escalate_to_legal if 90+ days overdue; otherwise flag_for_credit_hold.

USE 'wait' WHEN: the customer's historical avg_days_late is higher than the current days_overdue (they're not actually late by their own standards).

ALWAYS provide a clear 1-2 sentence reasoning that references specific facts from the customer's history.
"""


USER_PROMPT_TEMPLATE = """OVERDUE INVOICE:
- Invoice ID: {invoice_id}
- Customer: {customer_name} ({customer_id})
- Amount: ₹{amount:,.0f}
- Issue date: {issue_date}
- Due date: {due_date}
- Days overdue: {days_overdue}

CUSTOMER CONTEXT:
- Segment: {segment}
- Industry: {industry}
- Credit terms: {credit_terms} days
- Avg days late (historical): {avg_delay:.1f}
- % of invoices paid 30+ days late: {pct_severe:.1%}
- Currently open invoices for this customer: {num_open}
- Total open balance: ₹{open_balance:,.0f}
- Prior collection emails sent: {num_interactions} ({num_firm} firm/final-notice)
- Payment delay trend: {delay_trend:+.1f} days (recent 6mo vs prior 6mo — positive means getting worse)

ML PREDICTION:
- Probability this invoice will be 30+ days late: {late_prob:.0%}

Pick the best action and explain your reasoning."""


def _safe_format(template: str, **kwargs) -> str:
    """Format with None-tolerant defaults."""
    defaults = {
        "amount": 0, "days_overdue": 0, "avg_delay": 0.0, "pct_severe": 0.0,
        "num_open": 0, "open_balance": 0, "num_interactions": 0, "num_firm": 0,
        "delay_trend": 0.0, "late_prob": 0.0, "credit_terms": 30,
    }
    defaults.update({k: (v if v is not None else defaults.get(k, "")) for k, v in kwargs.items()})
    return template.format(**defaults)


class ActionPlanner:
    """Decides one action per overdue invoice using an LLM with structured tool calling."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def decide(self,
               invoice: dict,
               customer: dict,
               features: dict,
               segment: str,
               late_probability: float,
               trace=None) -> ActionDecision:
        """
        Decide what to do about this overdue invoice.
        Returns an ActionDecision with the chosen tool + arguments + reasoning.
        """
        user_msg = _safe_format(
            USER_PROMPT_TEMPLATE,
            invoice_id=invoice["invoice_id"],
            customer_name=customer["name"],
            customer_id=customer["customer_id"],
            amount=float(invoice["total_amount"]),
            issue_date=invoice["issue_date"],
            due_date=invoice["due_date"],
            days_overdue=int(features.get("days_overdue", 0)),
            segment=segment,
            industry=customer.get("industry", "Unknown"),
            credit_terms=customer["credit_terms_days"],
            avg_delay=float(features.get("customer_avg_days_late", 0.0)),
            pct_severe=float(features.get("customer_pct_severe_late", 0.0)),
            num_open=int(features.get("customer_num_open_invoices", 0)),
            open_balance=float(features.get("customer_open_balance", 0)),
            num_interactions=int(features.get("num_prior_interactions", 0)),
            num_firm=int(features.get("num_firm_interactions", 0)),
            delay_trend=float(features.get("customer_delay_trend", 0.0)),
            late_prob=float(late_probability),
        )

        # Call LLM with tools enabled
        response = self.llm.complete(
            system=SYSTEM_PROMPT,
            user=user_msg,
            tools=ACTION_TOOLS,
            tool_choice="required",     # force the model to pick a tool
            max_tokens=512,
        )

        decision = self._parse_response(response, invoice["invoice_id"])
        decision.confidence = float(late_probability)
        decision.customer_segment = segment
        decision.raw_llm_response = response.text

        if trace is not None:
            try:
                trace.update(
                    name="action_planner",
                    input={"invoice_id": invoice["invoice_id"], "segment": segment,
                           "late_probability": late_probability},
                    output=decision.to_dict(),
                )
            except Exception as e:
                logger.warning(f"Langfuse trace update failed: {e}")

        return decision

    def _parse_response(self, response, invoice_id: str) -> ActionDecision:
        """Extract the tool call from Groq's response."""
        msg = response.raw
        # Function-calling path (preferred)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            reasoning = args.pop("reasoning", "") or args.pop("reason", "")
            return ActionDecision(
                invoice_id=invoice_id,
                tool_name=tool_name,
                arguments=args,
                reasoning=reasoning,
            )

        # Fallback: extract tool call from text (last-resort heuristic)
        text = response.text or ""
        tool_match = re.search(r"(send_friendly_reminder|send_firm_escalation|propose_payment_plan|"
                                r"flag_for_credit_hold|escalate_to_legal|wait)", text)
        tool_name = tool_match.group(1) if tool_match else "send_friendly_reminder"
        return ActionDecision(
            invoice_id=invoice_id,
            tool_name=tool_name,
            arguments={},
            reasoning=text[:300],
        )


if __name__ == "__main__":
    # Sanity check (requires GROQ_API_KEY in env)
    import os
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in .env to test the agent")
        exit(0)

    planner = ActionPlanner()
    fake_invoice = {
        "invoice_id": "INV-2024-0042",
        "total_amount": 250000,
        "issue_date": "2024-03-15",
        "due_date": "2024-04-14",
    }
    fake_customer = {
        "customer_id": "CUST-007",
        "name": "Sharma Industries Pvt Ltd",
        "industry": "Manufacturing",
        "credit_terms_days": 30,
    }
    fake_features = {
        "days_overdue": 22,
        "customer_avg_days_late": 28,
        "customer_pct_severe_late": 0.45,
        "customer_num_open_invoices": 3,
        "customer_open_balance": 580000,
        "num_prior_interactions": 5,
        "num_firm_interactions": 2,
        "customer_delay_trend": 8.0,
    }
    decision = planner.decide(fake_invoice, fake_customer, fake_features,
                                segment="Chronic Late", late_probability=0.78)
    print(f"\nDecision: {decision.tool_name}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Arguments: {decision.arguments}")
