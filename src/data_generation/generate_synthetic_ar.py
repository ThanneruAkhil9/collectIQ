"""
Synthetic AR Dataset Generator
==============================
Creates a realistic 24-month AR dataset for an Indian mid-sized company:
- 50 customers across 5 personas (Strategic, Standard, Chronic Late, At Risk, Bad Debt)
- ~800 invoices with realistic payment delays per persona
- Matched payment records
- Prior collection interaction history per customer

Each customer's payment behavior is governed by their persona, ensuring
the segmentation model has a real signal to learn from and the XGBoost
classifier has predictable patterns.
"""
import json
import random
from datetime import datetime, timedelta, date
from pathlib import Path
from collections import defaultdict
from faker import Faker

fake = Faker('en_IN')
Faker.seed(42)
random.seed(42)

# ---------- Persona definitions ----------
# Each persona controls payment behavior parameters.
PERSONAS = {
    "Strategic": {
        "weight": 0.20,          # 20% of customers
        "delay_mean": 5,          # avg days late (or early if negative)
        "delay_std": 4,
        "default_rate": 0.005,    # 0.5% chance of write-off
        "monthly_volume": (8, 15),  # invoices/month range
        "invoice_amount": (200000, 1500000),  # high-value
        "industry_hint": ["IT", "Manufacturing", "Pharma"],
        "credit_limit": 50000000,
        "credit_terms_days": 45,
    },
    "Standard": {
        "weight": 0.45,
        "delay_mean": 8,
        "delay_std": 6,
        "default_rate": 0.01,
        "monthly_volume": (3, 8),
        "invoice_amount": (50000, 400000),
        "industry_hint": ["Retail", "Services", "Logistics"],
        "credit_limit": 5000000,
        "credit_terms_days": 30,
    },
    "Chronic Late": {
        "weight": 0.20,
        "delay_mean": 35,
        "delay_std": 12,
        "default_rate": 0.03,
        "monthly_volume": (2, 5),
        "invoice_amount": (30000, 200000),
        "industry_hint": ["Construction", "Wholesale", "Hospitality"],
        "credit_limit": 2000000,
        "credit_terms_days": 30,
    },
    "At Risk": {
        "weight": 0.10,
        "delay_mean": 55,           # gets worse over time (see deterioration)
        "delay_std": 18,
        "default_rate": 0.15,
        "monthly_volume": (1, 4),
        "invoice_amount": (40000, 300000),
        "industry_hint": ["Real Estate", "Textile", "Restaurants"],
        "credit_limit": 1500000,
        "credit_terms_days": 30,
        "deteriorating": True,      # delay grows month-over-month
    },
    "Bad Debt": {
        "weight": 0.05,
        "delay_mean": 90,
        "delay_std": 30,
        "default_rate": 0.40,       # 40% of invoices written off
        "monthly_volume": (1, 2),
        "invoice_amount": (20000, 150000),
        "industry_hint": ["Defunct startups", "Bankrupt entities", "Disputed accounts"],
        "credit_limit": 500000,
        "credit_terms_days": 30,
    },
}

INDIAN_STATES = [
    ("Maharashtra", "27"), ("Delhi", "07"), ("Karnataka", "29"),
    ("Tamil Nadu", "33"), ("Telangana", "36"), ("Gujarat", "24"),
    ("West Bengal", "19"), ("Uttar Pradesh", "09"), ("Haryana", "06"),
    ("Kerala", "32"), ("Punjab", "03"), ("Rajasthan", "08"),
]


def _generate_gstin(state_code: str) -> str:
    """Generate a syntactically valid GSTIN: 2-digit state + 10-char PAN + 1Z1."""
    import string
    pan = ''.join(random.choices(string.ascii_uppercase, k=5))
    pan += ''.join(random.choices(string.digits, k=4))
    pan += random.choice(string.ascii_uppercase)
    return f"{state_code}{pan}1Z{random.choice(string.digits)}"


def generate_customers(num_customers: int = 50) -> list[dict]:
    """Generate customer master data with persona assignments."""
    customers = []
    # Build weighted persona list
    persona_pool = []
    for name, cfg in PERSONAS.items():
        count = max(1, round(num_customers * cfg["weight"]))
        persona_pool.extend([name] * count)
    # Trim/pad to exact size
    while len(persona_pool) < num_customers:
        persona_pool.append("Standard")
    persona_pool = persona_pool[:num_customers]
    random.shuffle(persona_pool)

    for i, persona_name in enumerate(persona_pool, start=1):
        cfg = PERSONAS[persona_name]
        state, state_code = random.choice(INDIAN_STATES)
        company = fake.company()
        if random.random() < 0.4:
            suffix = random.choice([" Pvt Ltd", " Industries", " Solutions", " Enterprises", " Corp"])
            if not any(s in company for s in ["Ltd", "Industries", "Solutions"]):
                company += suffix
        contact_name = fake.name()
        customers.append({
            "customer_id": f"CUST-{i:03d}",
            "name": company,
            "industry": random.choice(cfg["industry_hint"]),
            "city": fake.city(),
            "state": state,
            "gstin": _generate_gstin(state_code),
            "contact_name": contact_name,
            "contact_email": f"{contact_name.lower().split()[0]}@{company.lower().replace(' ', '').replace(',', '')[:15]}.com",
            "credit_limit": cfg["credit_limit"],
            "credit_terms_days": cfg["credit_terms_days"],
            "persona_truth": persona_name,    # ground truth — model shouldn't see this directly
            "onboarded": (date.today() - timedelta(days=random.randint(400, 1200))).isoformat(),
        })
    return customers


def _payment_delay_for_invoice(persona_name: str, invoice_age_months: int) -> int:
    """
    Sample a payment delay (days vs due date) for a single invoice.
    Negative = paid early. Positive = paid late.
    Deteriorating personas get worse over time.
    """
    cfg = PERSONAS[persona_name]
    mean = cfg["delay_mean"]
    std = cfg["delay_std"]

    # Deteriorating customers: delay grows by ~1 day/month
    if cfg.get("deteriorating") and invoice_age_months < 24:
        # invoices closer to "now" (smaller age) get larger delays
        deterioration = max(0, 24 - invoice_age_months) * 1.0
        mean += deterioration

    # Sample from normal, floor at -10 (rare early payments)
    delay = int(random.gauss(mean, std))
    return max(-10, delay)


def generate_invoices_and_payments(customers: list[dict],
                                    months: int = 24,
                                    as_of: date = None) -> tuple[list[dict], list[dict]]:
    """
    Generate 24 months of historical invoices + matched payments.
    The last 1 month (most recent) leaves some invoices OPEN (unpaid) — these
    are the "overdue" invoices the agent operates on.
    """
    if as_of is None:
        as_of = date.today()

    invoices = []
    payments = []
    inv_seq = 0

    for customer in customers:
        persona_name = customer["persona_truth"]
        cfg = PERSONAS[persona_name]
        credit_terms = customer["credit_terms_days"]

        for month_back in range(months, 0, -1):
            month_start = as_of - timedelta(days=month_back * 30)
            volume = random.randint(*cfg["monthly_volume"])

            for _ in range(volume):
                inv_seq += 1
                issue_date = month_start + timedelta(days=random.randint(0, 28))
                due_date = issue_date + timedelta(days=credit_terms)
                amount = round(random.uniform(*cfg["invoice_amount"]), 2)

                # 18% GST (CGST+SGST if intra-state, else IGST — assume buyer in Telangana)
                tax_amount = round(amount * 0.18, 2)
                total_amount = round(amount + tax_amount, 2)

                # Sample delay
                delay = _payment_delay_for_invoice(persona_name, month_back)

                # Default check
                is_default = random.random() < cfg["default_rate"]

                invoice = {
                    "invoice_id": f"INV-{issue_date.year}-{inv_seq:04d}",
                    "customer_id": customer["customer_id"],
                    "issue_date": issue_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "subtotal": amount,
                    "tax": tax_amount,
                    "total_amount": total_amount,
                    "currency": "INR",
                    "status": "open",            # set below
                    "paid_date": None,
                    "days_late_at_payment": None,
                }

                # Determine current status
                if is_default and month_back <= 6:
                    # Recent default — still open and very overdue
                    invoice["status"] = "overdue"
                elif month_back == 1:
                    # Most recent month: about 50% are still open (currently overdue or aging)
                    if random.random() < 0.50:
                        invoice["status"] = "overdue" if (as_of - due_date).days > 0 else "open"
                    else:
                        # Paid recently
                        paid_date = due_date + timedelta(days=delay)
                        if paid_date <= as_of:
                            invoice["status"] = "paid"
                            invoice["paid_date"] = paid_date.isoformat()
                            invoice["days_late_at_payment"] = delay
                            payments.append({
                                "payment_id": f"PAY-{inv_seq:05d}",
                                "invoice_id": invoice["invoice_id"],
                                "customer_id": customer["customer_id"],
                                "amount": total_amount,
                                "payment_date": paid_date.isoformat(),
                                "days_late": delay,
                            })
                        else:
                            invoice["status"] = "overdue" if (as_of - due_date).days > 0 else "open"
                else:
                    # Older months: paid (unless default)
                    paid_date = due_date + timedelta(days=delay)
                    if not is_default:
                        invoice["status"] = "paid"
                        invoice["paid_date"] = paid_date.isoformat()
                        invoice["days_late_at_payment"] = delay
                        payments.append({
                            "payment_id": f"PAY-{inv_seq:05d}",
                            "invoice_id": invoice["invoice_id"],
                            "customer_id": customer["customer_id"],
                            "amount": total_amount,
                            "payment_date": paid_date.isoformat(),
                            "days_late": delay,
                        })

                invoices.append(invoice)

    return invoices, payments


def generate_interactions(customers: list[dict], invoices: list[dict]) -> list[dict]:
    """
    Generate prior collection email history per customer.
    Used by the agent to understand 'has this customer been chased before?'
    """
    interactions = []
    by_customer = defaultdict(list)
    for inv in invoices:
        by_customer[inv["customer_id"]].append(inv)

    for customer in customers:
        persona = customer["persona_truth"]
        cfg = PERSONAS[persona]
        # Chronic Late / At Risk / Bad Debt: more historical chasing
        baseline_chases = {"Strategic": 1, "Standard": 3, "Chronic Late": 12, "At Risk": 18, "Bad Debt": 25}
        num = random.randint(0, baseline_chases.get(persona, 3))
        customer_invoices = sorted(by_customer[customer["customer_id"]],
                                    key=lambda x: x["issue_date"])
        for _ in range(num):
            if not customer_invoices:
                break
            inv = random.choice(customer_invoices)
            interaction_date_str = inv["issue_date"]
            interaction_date = date.fromisoformat(interaction_date_str) + timedelta(
                days=random.randint(30, 90))
            tone = random.choices(
                ["friendly", "neutral", "firm", "final_notice"],
                weights={"Strategic": [80, 15, 5, 0], "Standard": [50, 35, 12, 3],
                         "Chronic Late": [10, 30, 50, 10], "At Risk": [5, 15, 50, 30],
                         "Bad Debt": [2, 8, 40, 50]}.get(persona, [50, 35, 12, 3]),
                k=1,
            )[0]
            interactions.append({
                "interaction_id": f"INT-{len(interactions) + 1:05d}",
                "customer_id": customer["customer_id"],
                "invoice_id": inv["invoice_id"],
                "date": interaction_date.isoformat(),
                "channel": random.choice(["email", "email", "email", "phone"]),
                "tone": tone,
                "outcome": random.choice(["no_response", "promised_payment", "paid",
                                          "disputed", "no_response", "no_response"]),
            })
    return interactions


def generate_all(num_customers: int = 50, months: int = 24,
                 output_dir: Path = None) -> dict:
    """Generate the complete synthetic dataset."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_customers} customers...")
    customers = generate_customers(num_customers)

    print(f"Generating {months} months of invoices + payments...")
    invoices, payments = generate_invoices_and_payments(customers, months=months)

    print(f"Generating interaction history...")
    interactions = generate_interactions(customers, invoices)

    # Write files
    with open(output_dir / "customers.json", "w") as f:
        json.dump(customers, f, indent=2, default=str)
    with open(output_dir / "invoices.json", "w") as f:
        json.dump(invoices, f, indent=2, default=str)
    with open(output_dir / "payments.json", "w") as f:
        json.dump(payments, f, indent=2, default=str)
    with open(output_dir / "interactions.json", "w") as f:
        json.dump(interactions, f, indent=2, default=str)

    open_count = sum(1 for inv in invoices if inv["status"] in ("open", "overdue"))
    overdue_count = sum(1 for inv in invoices if inv["status"] == "overdue")

    summary = {
        "customers": len(customers),
        "invoices": len(invoices),
        "payments": len(payments),
        "interactions": len(interactions),
        "open_invoices": open_count,
        "overdue_invoices": overdue_count,
        "by_persona": {p: sum(1 for c in customers if c["persona_truth"] == p)
                       for p in PERSONAS},
    }
    print("\n✅ Dataset generated:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    return summary


if __name__ == "__main__":
    generate_all()
