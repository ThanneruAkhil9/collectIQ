"""
CollectIQ — Gradio Live Demo
============================
Interactive UI showing:
  - Aging report (sortable by amount, days overdue, risk)
  - Per-invoice agent recommendation + reasoning
  - Drafted email with 3 tone variants
  - Outbound guardrail check
  - DSO projection if all recommendations accepted
"""
import os
import sys
import time
from pathlib import Path
from datetime import date
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


# ---- Lazy load pipeline once ----
print("Loading CollectIQ pipeline...")
from src.pipeline import CollectIQPipeline

# Build everything if not done
if not (Path("data/models/late_payment.xgb").exists() and
        Path("data/customers.json").exists()):
    print("First run — generating data + training models...")
    from setup_all import main as setup_main
    setup_main()

pipe = CollectIQPipeline(draft_emails=False)
print("✅ Pipeline ready")

BACKEND = "Groq (Llama 3.3 70B)" if os.getenv("GROQ_API_KEY") else "Dry-run (no LLM)"


# ---- Action tool icons / colors ----
ACTION_DISPLAY = {
    "send_friendly_reminder": ("🟢", "Friendly reminder", "#10b981"),
    "send_firm_escalation": ("🟠", "Firm escalation", "#f59e0b"),
    "propose_payment_plan": ("💡", "Payment plan", "#3b82f6"),
    "flag_for_credit_hold": ("🔴", "Credit hold", "#ef4444"),
    "escalate_to_legal": ("⚖️", "Legal escalation", "#7c3aed"),
    "wait": ("⏸️", "Wait", "#6b7280"),
}

SEGMENT_COLORS = {
    "Strategic": "#10b981", "Standard": "#3b82f6", "Chronic Late": "#f59e0b",
    "At Risk": "#ef4444", "Bad Debt": "#7c3aed",
}


def build_aging_table(max_rows: int = 50) -> pd.DataFrame:
    """Build the current overdue list as a table."""
    overdue = pipe.get_overdue_invoices()
    as_of = date.today()
    rows = []
    for inv in overdue:
        cust = pipe._customer_lookup[inv["customer_id"]]
        due = date.fromisoformat(inv["due_date"][:10])
        days_over = (as_of - due).days
        segment = pipe.customer_personas.get(cust["customer_id"], "Standard")
        rows.append({
            "Invoice": inv["invoice_id"],
            "Customer": cust["name"],
            "Segment": segment,
            "Amount (₹)": f"{float(inv['total_amount']):,.0f}",
            "Days Overdue": days_over,
        })
    df = pd.DataFrame(rows).sort_values("Days Overdue", ascending=False).head(max_rows)
    return df.reset_index(drop=True)


def analyze_invoice(invoice_id: str):
    """Run the agent on a single invoice and return formatted outputs."""
    if not invoice_id:
        return "Please enter an invoice ID.", "", "", "", ""

    invoice_id = invoice_id.strip().upper()
    invoice = next((inv for inv in pipe.invoices if inv["invoice_id"].upper() == invoice_id), None)
    if not invoice:
        return f"❌ Invoice {invoice_id} not found.", "", "", "", ""

    t0 = time.time()
    item = pipe.process_invoice(invoice, as_of=date.today(),
                                   draft_email=bool(pipe.drafter))
    elapsed = int((time.time() - t0) * 1000)

    cust = item.customer
    icon, label, color = ACTION_DISPLAY.get(item.decision.tool_name, ("•", "Action", "#374151"))
    seg_color = SEGMENT_COLORS.get(item.segment, "#374151")

    # Summary card
    summary_md = f"""
## {icon} Recommended action: **{label}**

| Field | Value |
|---|---|
| Invoice | `{invoice['invoice_id']}` |
| Customer | **{cust['name']}** ({cust['customer_id']}) |
| Industry | {cust.get('industry', '-')} |
| Amount | ₹{float(invoice['total_amount']):,.0f} |
| Days overdue | **{item.days_overdue}** |
| Segment | <span style='background:{seg_color}; color:white; padding:2px 8px; border-radius:4px'>{item.segment}</span> |
| P(30+ days late) | **{item.late_probability:.0%}** |

⏱ Analyzed in {elapsed} ms
"""

    # Reasoning
    reasoning_md = f"### 🧠 Agent reasoning\n\n> {item.decision.reasoning or '(no reasoning provided)'}\n\n"
    if item.decision.arguments:
        reasoning_md += "**Arguments:**\n"
        for k, v in item.decision.arguments.items():
            reasoning_md += f"- `{k}`: {v}\n"

    # Email
    email_md = ""
    if item.email:
        email_md = f"### ✉️  Drafted email ({item.email.tone})\n\n"
        email_md += f"**Subject:** {item.email.subject}\n\n```\n{item.email.body}\n```"
    elif item.decision.tool_name == "wait":
        email_md = "_No email — action is 'wait'._"
    else:
        email_md = "_Email drafting skipped (LLM not configured)._"

    # Guardrail report
    guard_md = ""
    if item.validation:
        guard_md = f"### 🛡  Outbound guardrails\n\n{item.validation.summary()}\n\n"
        for issue in item.validation.issues:
            emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
            guard_md += f"- {emoji} **{issue.category}**: {issue.detail}\n"
    elif item.email:
        guard_md = "_Guardrails skipped._"

    return summary_md, reasoning_md, email_md, guard_md, ""


def run_full_queue(max_invoices: int):
    """Process the top N overdue invoices and return a summary."""
    if not pipe.planner:
        return ("⚠️  GROQ_API_KEY not set — running in dry-run rule-based mode.\n\n"
                "For a real demo, set the key in `.env` and restart.", None, None)

    t0 = time.time()
    result = pipe.run(max_invoices=int(max_invoices), draft_emails=False)
    elapsed = int((time.time() - t0) * 1000)

    # Summary
    counts = defaultdict(int)
    for item in result.queue:
        counts[item.decision.tool_name] += 1

    md = f"## 📊 Pipeline run — {result.as_of} \n\n"
    md += f"Processed **{len(result.queue)} invoices** in {elapsed} ms.\n\n"
    md += "### Action breakdown\n\n"
    for tool, n in sorted(counts.items(), key=lambda x: -x[1]):
        icon, label, _ = ACTION_DISPLAY.get(tool, ("•", tool, ""))
        md += f"- {icon} **{label}**: {n}\n"

    # DSO box
    dso = result.dso_report
    dso_md = f"""
### 💰 Projected DSO impact

| Metric | Value |
|---|---:|
| Baseline DSO | **{dso.baseline_dso_days:.1f} days** |
| Projected DSO (post-actions) | **{dso.projected_dso_days:.1f} days** |
| DSO reduction | **−{dso.dso_reduction_days:.1f} days** |
| Baseline AR balance | ₹{dso.baseline_ar_balance:,.0f} |
| Projected AR balance | ₹{dso.projected_ar_balance:,.0f} |
| **Working capital unlocked** | **₹{dso.working_capital_unlocked:,.0f}** |
| Estimated annual revenue | ₹{dso.annual_revenue:,.0f} |
"""

    # Top 10 queue
    queue_rows = []
    for item in result.queue[:10]:
        icon, label, _ = ACTION_DISPLAY.get(item.decision.tool_name, ("•", item.decision.tool_name, ""))
        queue_rows.append({
            "Invoice": item.invoice["invoice_id"],
            "Customer": item.customer["name"][:35],
            "Segment": item.segment,
            "Days Over": item.days_overdue,
            "P(late)": f"{item.late_probability:.0%}",
            "Action": f"{icon} {label}",
            "Reasoning": item.decision.reasoning[:120] + ("..." if len(item.decision.reasoning) > 120 else ""),
        })
    queue_df = pd.DataFrame(queue_rows)

    return md + "\n\n" + dso_md, queue_df, None


# ============================================================
# UI
# ============================================================

CSS = """
.gradio-container { max-width: 1400px !important }
footer { display: none !important }
"""

INTRO = f"""
# 💰 CollectIQ — Autonomous AR Collections Agent

Autonomous agent for Accounts Receivable. Predicts which invoices will be late (**XGBoost**),
segments customers into 5 personas (**k-means**), and picks an action from 6 tools (**LLM agent**)
— all with outbound guardrails and DSO impact projection.

**LLM:** {BACKEND}  |  **Cost:** ₹0  |  **Code:** [github.com/ThanneruAkhil9/collectiq](https://github.com/ThanneruAkhil9)

Sister project to **[AP-hybrid-rag](https://github.com/ThanneruAkhil9/AP-hybrid-rag)** — together a complete Finance + AI portfolio.
"""

with gr.Blocks(title="CollectIQ — AR Collections Agent",
                theme=gr.themes.Soft(),
                css=CSS) as demo:
    gr.Markdown(INTRO)

    with gr.Tabs():
        # ============= Tab 1: Single invoice deep dive =============
        with gr.Tab("🔎 Analyze single invoice"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Pick or type an invoice ID")
                    invoice_input = gr.Textbox(label="Invoice ID",
                                                  placeholder="e.g., INV-2024-3691",
                                                  value="INV-2024-3691")
                    analyze_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")
                    gr.Markdown("### Currently overdue invoices")
                    aging_df = gr.Dataframe(value=build_aging_table(20), interactive=False,
                                              wrap=True)
                with gr.Column(scale=2):
                    summary_out = gr.Markdown()
                    reasoning_out = gr.Markdown()
                    email_out = gr.Markdown()
                    guard_out = gr.Markdown()

            analyze_btn.click(analyze_invoice, inputs=[invoice_input],
                              outputs=[summary_out, reasoning_out, email_out, guard_out, gr.State()])

        # ============= Tab 2: Full daily queue =============
        with gr.Tab("📋 Run daily queue"):
            gr.Markdown(
                "### Process the most overdue N invoices in a single batch\n"
                "Each invoice goes through: prediction → agent decision → DSO projection.\n"
                "*Email drafting disabled for batch mode to keep latency low.*"
            )
            with gr.Row():
                n_invoices = gr.Slider(1, 50, value=10, step=1, label="Number of invoices to process")
                run_btn = gr.Button("▶️  Run batch", variant="primary", size="lg")
            batch_summary = gr.Markdown()
            batch_queue = gr.Dataframe(label="Action queue (top 10)", interactive=False, wrap=True)

            run_btn.click(run_full_queue, inputs=[n_invoices],
                          outputs=[batch_summary, batch_queue, gr.State()])

        # ============= Tab 3: About =============
        with gr.Tab("ℹ️  About"):
            gr.Markdown(f"""
### How it works

1. **Customer Segmentation** (offline, nightly): k-means clusters all customers into
   5 personas based on payment behavior — Strategic, Standard, Chronic Late, At Risk, Bad Debt.
2. **Late-Payment Prediction** (per invoice): XGBoost classifier trained on 6,700 historical
   paid invoices. Predicts P(30+ days late) using 16 features. **F1 = 0.77, AUC = 0.98.**
3. **Action Planner** (LLM agent): Llama 3.3 70B (Groq) with 6 tools — friendly reminder,
   firm escalation, payment plan, credit hold, legal escalation, wait. Picks one action
   per invoice with reasoning.
4. **Tone-Matched Email Drafter**: generates customer-facing email conditioned on tone
   profile (friendly / neutral / firm / final notice) and segment.
5. **Outbound Guardrails**: regex validators check every drafted email — invoice numbers,
   amounts, customer names verified against source data. Blocks threats or false legal claims.
6. **DSO Simulator**: projects working capital unlocked if recommendations are accepted.

### Stack (100% free)

| Layer | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq free tier |
| ML | XGBoost + scikit-learn |
| Observability | Langfuse Cloud Hobby |
| UI | Gradio |
| Hosting | HuggingFace Spaces |

**Total: ₹0/month**
""")

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )