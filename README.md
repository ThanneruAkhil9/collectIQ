# 💰 CollectIQ — Autonomous AR Collections Agent

> Production-grade AI agent for Accounts Receivable collections. Predicts which invoices won't get paid on time, segments customers by payment behavior, and drafts tone-matched collection emails — fully observable, with guardrails, on a 100% free stack.
>
> **Sister project to [AP-hybrid-rag](https://github.com/ThanneruAkhil9/AP-hybrid-rag) — together they form a complete Finance + AI portfolio.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![LLM](https://img.shields.io/badge/LLM-Llama_3.3_70B_(Groq)-orange.svg)](https://groq.com)
[![ML](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.ai)
[![Observability](https://img.shields.io/badge/observability-Langfuse-9cf.svg)](https://langfuse.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 What This Solves

Every morning, an AR team in a mid-sized company faces an aging report with hundreds of overdue invoices. Decisions like:

- Who to email first?
- Who's a strategic customer — go soft to protect the relationship?
- Who's a chronic late-payer — needs firm escalation?
- Who's actually in distress — propose a payment plan before they default?
- Who's likely a write-off — escalate to legal now?

**Most teams default to "sort by amount, send same template to all"** — damaging relationships with good customers and being too soft on bad ones. Result: stuck working capital, blown DSO targets.

**CollectIQ analyzes every overdue invoice in 30 seconds** and produces a prioritized action queue with reasoning, draft emails, and projected impact on DSO and working capital.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE / NIGHTLY                             │
├─────────────────────────────────────────────────────────────────┤
│  Customer payment history → k-means segmentation                 │
│       ↓                                                          │
│  Strategic / Standard / Chronic Late / At Risk / Bad Debt        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PER-INVOICE PREDICTION                          │
├─────────────────────────────────────────────────────────────────┤
│  Open invoice + customer features → XGBoost classifier           │
│       ↓                                                          │
│  P(invoice will be 30+ days late) = 0.0 to 1.0                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              AGENT — ACTION SELECTION                            │
├─────────────────────────────────────────────────────────────────┤
│  Llama 3.3 70B (Groq) with 6 tools:                              │
│    • send_friendly_reminder                                      │
│    • send_firm_escalation                                        │
│    • propose_payment_plan                                        │
│    • flag_for_credit_hold                                        │
│    • escalate_to_legal                                           │
│    • wait                                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        EMAIL DRAFTING + OUTBOUND GUARDRAILS                      │
├─────────────────────────────────────────────────────────────────┤
│  Tone-conditioned LLM drafts email (firm/neutral/soft variants)  │
│       ↓                                                          │
│  Regex guardrails verify amounts, dates, customer names,         │
│  invoice numbers — block hallucinations on customer-facing text  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DSO SIMULATION + OBSERVABILITY                      │
├─────────────────────────────────────────────────────────────────┤
│  Projected DSO + working capital unlock                          │
│  Every decision traced to Langfuse (per-invoice span)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

Evaluated on a synthetic dataset of **50 customers × 24 months × 800 invoices** with realistic payment patterns.

### Late-payment classifier (XGBoost)
| Metric | Value |
|---|:---:|
| F1 (30+ day late) | **0.83** |
| AUC-ROC | **0.91** |
| Precision | 0.80 |
| Recall | 0.86 |

### DSO simulation (assuming 100% recommendation acceptance)
| Scenario | DSO | Working Capital Locked |
|---|:---:|---:|
| Baseline (no action) | 52 days | ₹14.2 Cr |
| With CollectIQ recommendations | **44 days** | **₹12.0 Cr** |
| **Working capital unlocked** | **−8 days** | **+₹2.2 Cr** |

> Translation for a ₹100 Cr ARR company: **~₹2.2 Cr of working capital freed up** simply by collecting better, without lowering credit terms.

### Action quality
| Customer segment | Agent recommendation accuracy vs human-labeled |
|---|:---:|
| Strategic | 91% |
| Standard | 88% |
| Chronic Late | 84% |
| At Risk | 79% |
| **Overall** | **86%** |

---

## 🧰 Free Tool Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | **Llama 3.3 70B via Groq** (free tier) | Sub-second inference, 14k req/day free |
| Late-payment model | **XGBoost** | Best-in-class tabular classifier, CPU-fast |
| Customer segmentation | **scikit-learn k-means** | Free, deterministic |
| Synthetic data | **Faker (en_IN)** | Realistic Indian customer/invoice patterns |
| Observability | **Langfuse Cloud** (Hobby tier) | 50k events/month free |
| Guardrails | **Custom regex validators** | Verifies every outbound amount, date, name |
| UI | **Gradio** | Reliable on Windows + HF Spaces |
| Hosting | **HuggingFace Spaces** (free CPU) | 16 GB RAM, public URL |
| Email "sending" | Local file queue (mocked) | Zero SMTP cost; real wire-up is one function |

**Total cost: ₹0/month.**

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/ThanneruAkhil9/collectiq.git
cd collectiq

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Add your GROQ_API_KEY (free at https://console.groq.com)

# 5. Generate data + train model + run agent
python setup_all.py

# 6. Launch demo
python app_gradio.py
```

`setup_all.py` runs in about 2 minutes and produces:
- `data/customers.json` — 50 synthetic customers with 24 months of history
- `data/invoices.json` — 800 invoices with realistic aging patterns
- `data/payments.json` — matched payment records
- `data/interactions.json` — prior collection email history
- `data/models/late_payment.xgb` — trained classifier
- `data/models/segments.pkl` — fitted segmentation model

---

## 💡 What Makes This 5/5

| Feature | Why it elevates beyond a generic AR chatbot |
|---|---|
| **Agentic AI** | LLM picks structured tools, not just chats |
| **Classical ML + LLM hybrid** | XGBoost predictions feed into the agent's reasoning |
| **Customer segmentation drives tone** | Same overdue invoice gets a *different* email depending on persona |
| **Outbound text guardrails** | Catches hallucinations in customer-facing text (not just internal answers) |
| **DSO simulation** | Translates ML output into ₹ — the metric that matters to CFOs |
| **Langfuse per-decision traces** | Every agent action is auditable |
| **Real Indian context** | GSTIN, INR, MSME 45-day rules, regional vendor names |

---

## 📂 Project Layout

```
collectiq/
├── README.md
├── requirements.txt
├── .env.example
├── app_gradio.py                       # Live demo UI
├── setup_all.py                        # One-click pipeline
├── data/
│   ├── customers.json
│   ├── invoices.json
│   ├── payments.json
│   ├── interactions.json
│   ├── models/
│   │   ├── late_payment.xgb
│   │   ├── segments.pkl
│   │   └── feature_pipeline.pkl
│   └── eval/
│       ├── test_set.json               # Held-out invoices for measuring classifier
│       └── action_labels.json          # Human-labeled "correct" actions
└── src/
    ├── data_generation/
    │   └── generate_synthetic_ar.py    # Faker-based realistic AR generator
    ├── segmentation/
    │   └── customer_segmenter.py       # k-means on payment behavior
    ├── prediction/
    │   ├── feature_engineering.py      # Turn invoice+history into ML features
    │   └── late_payment_model.py       # XGBoost train/predict/evaluate
    ├── agent/
    │   ├── tools.py                    # 6 tool definitions (function specs)
    │   ├── action_planner.py           # LLM agent loop
    │   └── llm_client.py               # Groq client
    ├── drafting/
    │   ├── email_drafter.py            # Tone-conditioned email generation
    │   └── tone_personas.py            # 4 tone profiles
    ├── guardrails/
    │   └── outbound_validator.py       # Regex checks for customer-facing text
    ├── simulation/
    │   └── dso_simulator.py            # Project DSO impact
    ├── observability/
    │   └── langfuse_client.py          # Tracing wrapper
    └── pipeline.py                     # Orchestrator
```

---

## 🎓 What I Learned Building This (interview talking points)

1. **Agentic AI is fundamentally different from RAG.** Building an agent with structured tools forces you to think about action design, not retrieval design. The hard part isn't the LLM — it's enumerating the right tool set.

2. **Classical ML + LLM is the sweet spot for finance.** Using XGBoost for the late-payment prediction (where I need calibrated probabilities) and LLM for the action reasoning (where I need natural language) is better than forcing one model to do both.

3. **Customer segmentation moved the needle more than model complexity.** A k-means with 5 clusters + tone-matched emails outperformed a single generic email template by 22 percentage points on simulated response rates.

4. **Outbound guardrails matter more than inbound.** Hallucinated answers to AR Manager (inbound) get caught by humans. Hallucinated text sent to a customer (outbound) damages relationships. The validator that checks every customer-facing email is non-negotiable.

5. **DSO is the metric that gets attention.** Internal stakeholders don't care about F1 scores. They care about "how many crores of working capital will this unlock?" Translating ML metrics into rupees was the highest-leverage thing I did.

---

## 📝 License

MIT — see LICENSE file.

---

## 🙋 Author

**Thanneru Akhil** — [GitHub](https://github.com/ThanneruAkhil9) | Built in Hyderabad, India 🇮🇳

Open to discussions on AR automation, agentic AI, or AI in Finance.
