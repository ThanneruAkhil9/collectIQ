"""
Email Drafter
=============
Generates customer-facing collection emails conditioned on:
  - Action type (chosen by the agent)
  - Customer segment + history
  - Tone profile (firm / neutral / friendly / final_notice)

Optionally generates 3 variants (firm/neutral/soft) so the AR manager
can pick the right one.
"""
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from ..agent.llm_client import LLMClient
from .tone_personas import TONE_PROFILES, get_tone_for_action


@dataclass
class DraftedEmail:
    invoice_id: str
    tone: str
    subject: str
    body: str
    raw_llm_output: str = ""

    def to_dict(self) -> dict:
        return {
            "invoice_id": self.invoice_id,
            "tone": self.tone,
            "subject": self.subject,
            "body": self.body,
        }


SYSTEM_PROMPT = """You are a professional AR collections specialist writing customer-facing emails in Indian English.

Your goal: draft a clear, professional email that achieves the action's goal (reminder, escalation, payment plan) while preserving the business relationship.

RULES:
1. Use ONLY facts provided to you. Never invent invoice numbers, amounts, or dates.
2. Indian context: write in Indian English. Use ₹ for currency. Reference Indian business norms (Net 30, MSME 45-day rules where relevant).
3. Be respectful but appropriately firm based on the tone instruction.
4. NEVER threaten — only state factual consequences (credit hold, interest charges, legal escalation).
5. Keep it within the word budget.
6. Output format: first line = subject. Then blank line. Then body. Nothing else.
"""


USER_TEMPLATE = """ACTION: {action}
TONE: {tone} — {tone_description}
TONE INSTRUCTION: {tone_instruction}
MAX WORDS: {max_words}

INVOICE FACTS (use only these — do not invent):
- Invoice ID: {invoice_id}
- Amount: ₹{amount:,.0f}
- Issue date: {issue_date}
- Due date: {due_date}
- Days overdue: {days_overdue}

CUSTOMER:
- Company: {company_name}
- Contact: {contact_name}
- Segment: {segment}

CONTEXT:
- Avg days late historically: {avg_delay:.0f}
- Prior collection emails sent: {num_interactions}
- Currently {num_open} open invoices, total ₹{open_balance:,.0f}

{extra_instructions}

Draft the email now. Use the salutation: "{salutation}"
"""


class EmailDrafter:
    def __init__(self, llm_client: Optional[LLMClient] = None,
                 sender_name: str = "Akhil — AR Team"):
        self.llm = llm_client or LLMClient()
        self.sender_name = sender_name

    def draft(self,
              invoice: dict,
              customer: dict,
              features: dict,
              action: str,
              segment: str,
              tone: Optional[str] = None,
              extra_instructions: str = "",
              trace=None) -> DraftedEmail:
        """Draft a single email for the given action + tone."""
        if tone is None:
            tone = get_tone_for_action(action, segment)
        if tone is None:
            raise ValueError(f"No tone defined for action '{action}'")

        profile = TONE_PROFILES[tone]

        # Pick contact first name; fall back to "team" if missing
        contact = customer.get("contact_name", "")
        contact_first = contact.split()[0] if contact else "team"
        company = customer.get("name", "your company")

        salutation = profile["salutation"].format(contact_first_name=contact_first)

        user_msg = USER_TEMPLATE.format(
            action=action,
            tone=tone,
            tone_description=profile["description"],
            tone_instruction=profile["llm_instruction"],
            max_words=profile["max_words"],
            invoice_id=invoice["invoice_id"],
            amount=float(invoice["total_amount"]),
            issue_date=invoice["issue_date"],
            due_date=invoice["due_date"],
            days_overdue=int(features.get("days_overdue", 0)),
            company_name=company,
            contact_name=contact or "Sir/Madam",
            segment=segment,
            avg_delay=float(features.get("customer_avg_days_late", 0.0)),
            num_interactions=int(features.get("num_prior_interactions", 0)),
            num_open=int(features.get("customer_num_open_invoices", 0)),
            open_balance=float(features.get("customer_open_balance", 0)),
            extra_instructions=extra_instructions,
            salutation=salutation,
        )

        response = self.llm.complete(
            system=SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=400,
        )

        subject, body = self._parse_email(response.text, fallback_invoice=invoice["invoice_id"])
        body = self._append_closing(body, profile["closing"])

        email = DraftedEmail(
            invoice_id=invoice["invoice_id"],
            tone=tone,
            subject=subject,
            body=body,
            raw_llm_output=response.text,
        )

        if trace is not None:
            try:
                trace.update(name="email_drafter", input={"tone": tone, "action": action},
                              output=email.to_dict())
            except Exception as e:
                logger.warning(f"Langfuse trace failed: {e}")

        return email

    def draft_three_variants(self, invoice, customer, features, action, segment, trace=None):
        """Generate firm/neutral/soft variants for AR manager to pick from."""
        variants = {}
        default_tone = get_tone_for_action(action, segment)
        tones_to_try = self._three_tones(default_tone)
        for tone in tones_to_try:
            try:
                variants[tone] = self.draft(invoice, customer, features, action,
                                              segment, tone=tone, trace=trace)
            except Exception as e:
                logger.warning(f"Failed to draft {tone} variant: {e}")
        return variants

    def _three_tones(self, default: str) -> list[str]:
        ordering = ["friendly", "neutral", "firm", "final_notice"]
        if default is None or default not in ordering:
            return ["friendly", "neutral", "firm"]
        idx = ordering.index(default)
        lo = max(0, idx - 1)
        hi = min(len(ordering), idx + 2)
        return ordering[lo:hi]

    def _parse_email(self, raw: str, fallback_invoice: str) -> tuple[str, str]:
        """Split LLM output into subject and body."""
        text = raw.strip()
        if "\n" not in text:
            return f"Re: Invoice {fallback_invoice}", text
        lines = text.split("\n", 1)
        subject_line = lines[0].strip()
        # Strip "Subject:" prefix if model adds it
        if subject_line.lower().startswith("subject:"):
            subject_line = subject_line[8:].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        # If body is empty (model only emitted subject), keep something usable
        if not body:
            body = text
        return subject_line, body

    def _append_closing(self, body: str, closing_template: str) -> str:
        """Append the closing template if not already present."""
        closing = closing_template.format(sender_name=self.sender_name)
        # If body already contains "Regards" or "Best," etc, don't double-up
        last_300 = body[-300:].lower()
        if any(token in last_300 for token in ["regards,", "best,", "sincerely,", "thanks,"]):
            return body
        return body.rstrip() + "\n\n" + closing
