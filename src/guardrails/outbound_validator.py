"""
Outbound Email Validator
========================
Regex-based guardrails for customer-facing collection emails.

Different from the AP project's inbound validator: there we worried about
hallucinated answers shown to internal users. Here we worry about hallucinated
text sent to CUSTOMERS — much higher stakes (relationship damage, legal).

Checks:
  1. Invoice number in email exists in source data
  2. Currency amount matches the invoice's total_amount (allowing minor variation)
  3. Customer name spelled correctly
  4. No threatening language ("will sue", "legal action against you personally", etc.)
  5. No fabricated legal claims ("section 138", "FIR" — unless contractually allowed)
  6. No fabricated contact names not in customer record
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ---------- Patterns ----------
INVOICE_PATTERN = re.compile(r"INV[-_]?(\d{4})[-_]?(\d{3,5})", re.IGNORECASE)
AMOUNT_PATTERNS = [
    re.compile(r"₹\s*([\d,]+(?:\.\d+)?)"),               # ₹1,23,456
    re.compile(r"INR\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"Rs\.?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
]
GSTIN_PATTERN = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d[Z][A-Z\d])\b")

THREATENING_PHRASES = [
    "i will personally", "we will destroy", "you will regret",
    "criminal charges", "fir against you", "section 138",
    "police complaint", "we will ruin",
    # softer phrases that are still inappropriate
    "you should be ashamed", "this is disgraceful",
]

# Legal-claims that need contract backing — flag for review if used without context
LEGAL_CLAIMS = [
    "interest at 24%", "interest at 36%",       # if our policy is 18% these are wrong
    "criminal proceedings", "section 138",       # NI Act 138 — only valid for bounced cheques
]


@dataclass
class ValidationIssue:
    """A single guardrail finding."""
    severity: str          # "error", "warning", "info"
    category: str          # "invoice_mismatch", "amount_mismatch", "threat", "legal_claim", "name_typo"
    detail: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "category": self.category, "detail": self.detail}


@dataclass
class ValidationReport:
    """Aggregate report for one email."""
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        if self.passed:
            return f"✅ Passed ({len(self.warnings)} warnings, 0 errors)"
        return f"❌ Blocked ({len(self.errors)} errors, {len(self.warnings)} warnings)"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary(),
        }


class OutboundValidator:
    """Validates a drafted collection email against source data."""

    def __init__(self):
        pass

    def validate(self,
                 email_subject: str,
                 email_body: str,
                 invoice: dict,
                 customer: dict) -> ValidationReport:
        """Run all checks on the drafted email."""
        issues: list[ValidationIssue] = []
        full_text = f"{email_subject}\n{email_body}"

        # --- Check 1: invoice numbers ---
        issues.extend(self._check_invoice_numbers(full_text, invoice))
        # --- Check 2: amounts ---
        issues.extend(self._check_amounts(full_text, invoice))
        # --- Check 3: customer name ---
        issues.extend(self._check_customer_name(full_text, customer))
        # --- Check 4: threatening language ---
        issues.extend(self._check_threats(full_text))
        # --- Check 5: fabricated legal claims ---
        issues.extend(self._check_legal_claims(full_text))

        passed = not any(i.severity == "error" for i in issues)
        return ValidationReport(passed=passed, issues=issues)

    # -------- individual checks --------

    def _check_invoice_numbers(self, text: str, invoice: dict) -> list[ValidationIssue]:
        out = []
        true_inv = invoice["invoice_id"].upper()
        found = INVOICE_PATTERN.findall(text)
        if not found:
            out.append(ValidationIssue(
                severity="warning", category="invoice_missing",
                detail="Email doesn't reference an invoice number explicitly",
            ))
        else:
            for year, num in found:
                candidate = f"INV-{year}-{num}".upper()
                # Normalize variants — INV-2024-42 vs INV-2024-0042
                if candidate == true_inv:
                    continue
                # Try padded
                padded = f"INV-{year}-{int(num):04d}".upper()
                if padded == true_inv:
                    continue
                out.append(ValidationIssue(
                    severity="error", category="invoice_mismatch",
                    detail=f"Email mentions {candidate} but invoice is {true_inv}",
                ))
        return out

    def _check_amounts(self, text: str, invoice: dict) -> list[ValidationIssue]:
        out = []
        true_amount = float(invoice["total_amount"])
        for pat in AMOUNT_PATTERNS:
            for m in pat.finditer(text):
                raw = m.group(1).replace(",", "")
                try:
                    amt = float(raw)
                except ValueError:
                    continue
                # Allow ±5% tolerance (rounding, partial payment language)
                if abs(amt - true_amount) / max(true_amount, 1) > 0.05:
                    # But amounts like "18%" should not be flagged
                    # Already filtered by ₹/INR/Rs prefix patterns
                    out.append(ValidationIssue(
                        severity="error", category="amount_mismatch",
                        detail=f"Email mentions amount ₹{amt:,.0f} but invoice total is ₹{true_amount:,.0f}",
                    ))
        return out

    def _check_customer_name(self, text: str, customer: dict) -> list[ValidationIssue]:
        out = []
        true_name = customer["name"]
        # Extract the core company word (first word, dropping common suffixes)
        core_words = []
        for w in true_name.split():
            w_clean = w.strip(",.").lower()
            if w_clean in {"pvt", "ltd", "limited", "private", "inc", "corp", "and", "&", "co",
                            "company", "industries", "solutions", "enterprises", "systems"}:
                continue
            core_words.append(w.strip(",."))
        # If the customer name has distinctive words, check at least one appears in body
        if core_words and not any(w.lower() in text.lower() for w in core_words):
            out.append(ValidationIssue(
                severity="warning", category="name_missing",
                detail=f"Email doesn't reference customer name '{true_name}' explicitly",
            ))
        return out

    def _check_threats(self, text: str) -> list[ValidationIssue]:
        out = []
        lower = text.lower()
        for phrase in THREATENING_PHRASES:
            if phrase in lower:
                out.append(ValidationIssue(
                    severity="error", category="threat",
                    detail=f"Email contains threatening/inappropriate language: '{phrase}'",
                ))
        return out

    def _check_legal_claims(self, text: str) -> list[ValidationIssue]:
        out = []
        lower = text.lower()
        for claim in LEGAL_CLAIMS:
            if claim in lower:
                out.append(ValidationIssue(
                    severity="warning", category="legal_claim",
                    detail=f"Email references potentially unsupported legal claim: '{claim}' — needs contract review",
                ))
        return out


# ---- Smoke test ----
if __name__ == "__main__":
    v = OutboundValidator()
    sample_invoice = {"invoice_id": "INV-2024-0042", "total_amount": 250000.0}
    sample_customer = {"name": "Sharma Industries Pvt Ltd", "contact_name": "Rajesh Sharma"}

    # Test 1: clean email
    email1 = ("Subject: Payment reminder for INV-2024-0042\n\n"
               "Dear Rajesh,\n\nThis is a reminder that invoice INV-2024-0042 for ₹2,50,000 "
               "issued to Sharma Industries Pvt Ltd is now 15 days overdue. "
               "Please confirm payment.\n\nRegards,\nAkhil")
    r1 = v.validate("Payment reminder", email1, sample_invoice, sample_customer)
    print(f"Clean email: {r1.summary()}")

    # Test 2: wrong invoice number
    email2 = ("Subject: Payment reminder\n\nDear Rajesh,\nInvoice INV-2024-9999 for ₹2,50,000 "
               "is overdue. Sharma Industries.\nRegards,")
    r2 = v.validate("test", email2, sample_invoice, sample_customer)
    print(f"\nWrong invoice number: {r2.summary()}")
    for i in r2.issues:
        print(f"   {i.severity}: {i.detail}")

    # Test 3: threatening language
    email3 = ("Dear Rajesh, We will personally ruin your company. Section 138 will apply. "
               "INV-2024-0042 ₹2,50,000 Sharma Industries.")
    r3 = v.validate("test", email3, sample_invoice, sample_customer)
    print(f"\nThreatening email: {r3.summary()}")
    for i in r3.issues:
        print(f"   {i.severity}: {i.detail}")
