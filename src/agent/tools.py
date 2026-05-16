"""
Agent Tools
===========
Defines the 6 actions the LLM agent can choose from. Each tool has a JSON
schema (for the LLM function-calling interface) and a Python dataclass
(for typed downstream handling).

Designed to be enumerated cleanly so the agent's decision space is small,
auditable, and easy to evaluate.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal


# JSON schemas for Groq / OpenAI function calling format
ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_friendly_reminder",
            "description": (
                "Send a polite, low-pressure reminder email. Use for strategic or "
                "standard customers who are mildly overdue (typically 1-15 days). "
                "Preserves the relationship while nudging."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string", "description": "The invoice to remind about"},
                    "reasoning": {"type": "string", "description": "Why this is the right action (1-2 sentences)"},
                },
                "required": ["invoice_id", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_firm_escalation",
            "description": (
                "Send a firm escalation email. Use for chronic late payers or invoices "
                "21+ days overdue from non-strategic customers. Optionally CCs the customer's "
                "finance manager to add pressure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "cc_manager": {"type": "boolean", "description": "Whether to CC their finance manager"},
                    "reasoning": {"type": "string"},
                },
                "required": ["invoice_id", "cc_manager", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_payment_plan",
            "description": (
                "Propose a structured payment plan with installments. Use when customer is "
                "in distress (e.g., 'At Risk' segment, deteriorating payment trend) — recovers "
                "some cash rather than risking total non-payment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "num_installments": {"type": "integer", "description": "Number of monthly installments (typically 2-6)"},
                    "monthly_amount": {"type": "number", "description": "Amount per installment in INR"},
                    "reasoning": {"type": "string"},
                },
                "required": ["invoice_id", "num_installments", "monthly_amount", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_credit_hold",
            "description": (
                "Place this customer on credit hold — no new orders/invoices until current "
                "balance is cleared. Use for severely delinquent or rapidly deteriorating customers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Why credit hold is justified"},
                },
                "required": ["invoice_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_legal",
            "description": (
                "Escalate this invoice to legal collections. Use only for invoices 90+ days "
                "overdue from customers in 'Bad Debt' segment or after exhausting other options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["invoice_id", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "Take no action for now. Use when the customer's historical pattern shows "
                "they always pay around a specific day (e.g., always pays on day 45) and we "
                "haven't yet reached that point. Avoids unnecessary pressure on reliable late-payers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "wait_days": {"type": "integer", "description": "How long to wait before re-evaluating"},
                    "reasoning": {"type": "string"},
                },
                "required": ["invoice_id", "wait_days", "reasoning"],
            },
        },
    },
]


# Tool names enum
ToolName = Literal[
    "send_friendly_reminder",
    "send_firm_escalation",
    "propose_payment_plan",
    "flag_for_credit_hold",
    "escalate_to_legal",
    "wait",
]


@dataclass
class ActionDecision:
    """Structured agent output. What the agent decided and why."""
    invoice_id: str
    tool_name: str
    arguments: dict = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0     # from XGBoost late-payment prediction
    customer_segment: str = ""
    raw_llm_response: str = ""

    def to_dict(self) -> dict:
        return {
            "invoice_id": self.invoice_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "customer_segment": self.customer_segment,
        }
