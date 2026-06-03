# WISDOM-PM — Intelligent Portfolio Management System

A production-grade, multi-agent Python system that implements the full 3-step WISDOM framework for disciplined Indian equity portfolio management.

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────┐
                        │          WISDOM-PM PIPELINE             │
                        └─────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
   ┌──────▼──────┐               ┌────────▼───────┐              ┌───────▼──────┐
   │  STEP 1     │               │  STEP 2        │              │  STEP 3      │
   │  Investor   │               │  WISDOM        │              │  Architecture│
   │  Profiling  │               │  Decision      │              │  & HITL      │
   │  & Bias     │               │  Matrix        │              │  Approval    │
   └─────────────┘               └────────────────┘              └──────────────┘

   ─────────────────────────── 4-Agent Pipeline ─────────────────────────────────

   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
   │  AGENT 1      │  │  AGENT 2      │  │  AGENT 3      │  │  AGENT 4      │
   │  Quant        │─▶│  Qualitative  │─▶│  Risk         │─▶│  Portfolio    │
   │  Analyst      │  │  Researcher   │  │  Manager      │  │  Manager      │
   │               │  │               │  │               │  │               │
   │ Tool-Calling  │  │ RAG + LLM     │  │ Rule-based    │  │ LLM Synthesis │
   │ Python/Pandas │  │ ChromaDB      │  │ Pure Python   │  │ Trade Memos   │
   └───────────────┘  └───────────────┘  └───────────────┘  └───────▼───────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │   HUMAN     │
                                                              │   SIGN-OFF  │
                                                              │   (HITL)    │
                                                              └─────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd wisdom_pm
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add:  ANTHROPIC_API_KEY=your_key_here
```

> **Without an API key**, the system runs in offline mode using rule-based agents and fallback market data. All 3 steps still execute fully.

### 3. Run the full pipeline

```bash
python main.py run
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py run` | Full 3-step pipeline with HITL approval |
| `python main.py run --macro` | Simulate macro shock (tests anti-panic locks) |
| `python main.py run --no-hitl` | Non-interactive run (no approval prompts) |
| `python main.py run --export` | Run pipeline + export JSON report |
| `python main.py score AMBER` | Score a single stock |
| `python main.py score ZEEL --macro` | Score with macro shock flag |
| `python main.py step1` | Show trade history + bias profile only |
| `python main.py arch` | Print architecture diagram |
| `python main.py export-report` | Export full JSON report |

---

## Project Structure

```
wisdom_pm/
│
├── main.py                  # CLI entry point (Typer)
├── orchestrator.py          # Full pipeline orchestrator
├── config.py                # Thresholds, tickers, bias profiles, sample data
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── quant_analyst.py     # Agent 1: Numerical analysis (Tool-Calling)
│   ├── qual_researcher.py   # Agent 2: RAG + LLM sentiment analysis
│   ├── risk_manager.py      # Agent 3: Portfolio risk flags (rule-based)
│   └── portfolio_manager.py # Agent 4: Memo synthesis + HITL orchestration
│
├── data/
│   └── market_data.py       # Yahoo Finance / fallback fundamentals fetcher
│
├── rag/
│   └── vector_store.py      # ChromaDB vector store + keyword fallback
│
├── scoring/
│   └── wisdom_scorer.py     # WISDOM 5-principle scorer (pure Python)
│
└── dashboard/
    └── terminal_ui.py       # Rich terminal dashboard
```

---

## The 3-Step Framework

### Step 1 — Historical Trade Analysis & Investor Profiling
Analyses 10 years of trade history to identify:
- **Strengths**: Structural theme identification (AMBER)
- **Bias 1 — Macro Panic**: Panic-sold WELSPUN at ₹21 (–70%) during Covid
- **Bias 2 — Sunk Cost**: Averaged down ZEEL through structural decline
- **Bias 3 — Cyclical Trap**: Misread DBL order-book peaks as durable growth

### Step 2 — WISDOM Decision Matrix
Scores every stock on 10 points across 5 principles:

| Principle | Quant Threshold | Qualitative Check |
|-----------|----------------|-------------------|
| 1. Business Quality | ROCE > 15%, FCF Yield > 5% | Analyst moat commentary |
| 2. Skin in the Game | Promoter > 40%, Pledged < 5% | Insider buys, capital allocation |
| 3. Reinvestment Runway | Retention > 60%, CapEx trending up | Industry tailwinds |
| 4. Structural vs Cyclical | Revenue Beta < 1.2 | LLM cyclicality detection |
| 5. Balance Sheet | D/E < 0.5, IC > 4x | Off-balance-sheet risks |

**Signal triggers:**

| Signal | Condition |
|--------|-----------|
| **BUY** | WISDOM Score > 8.0 AND PEG < 1.5 |
| **HOLD** | WISDOM Score > 7.0 (thesis intact) |
| **HOLD (Anti-Panic)** | WISDOM > 7.0 AND price drop > 20% due to macro only |
| **SELL** | ROCE < 12% for 2 consecutive quarters OR thesis break detected |
| **WATCH** | WISDOM Score 5.5 – 7.0 |

### Step 3 — Architecture & Trade Memos
- Multi-agent synthesis → Trade Recommendation Memo per stock
- **Human-in-the-Loop**: No trade auto-executes — fund manager must approve/reject via CLI
- LLM hallucination guardrail: LLM never computes numbers — only calls Python tools

---

## Hallucination Guardrails

```python
# ✅ CORRECT — LLM calls a tool, Python computes the number
{"name": "compute_wisdom_score", "input": {"ticker": "AMBER"}}
# Python returns: {"total_score": 9.1, "signal": "BUY"}

# ❌ NEVER — LLM asked to compute ROCE, CAGR, or weights itself
```

---

## Data Privacy

| Data type | Location | Sent to LLM? |
|-----------|----------|-------------|
| 10-year trade history | Encrypted PostgreSQL (on-premise) | **Never** |
| AUM amounts | Local config | **Never** |
| Analyst report text | ChromaDB vector store | **Anonymised snippets only** |
| Market prices | Yahoo Finance API | Not needed |

---

## Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...          # Required for LLM features; optional for offline mode
```

---

## Sample Output

```
╭─────────────────────────────────────────╮
│          WISDOM–PM                      │
│  Intelligent Portfolio Management       │
╰─────────────────────────────────────────╯

══ STEP 1 — Historical Trade Analysis ══
  AMBER   BUY  2017  ₹835   Structural EMS megatrend
  WELSPUN SELL 2020  ₹21    PANIC SELL [BIAS: MACRO_PANIC]
  ZEEL    BUY  2020  ₹175   Averaging down [BIAS: SUNK_COST]

══ STEP 2 — WISDOM Decision Matrix ══
  AMBER    9.1/10  → BUY
  WELSPUN  7.8/10  → HOLD
  ZEEL     3.9/10  → SELL  (ROCE 8.3% < 12% threshold)
  DBL      5.2/10  → WATCH (D/E 1.12, Cyclical trap)

══ STEP 3 — Trade Memos (HITL) ══
  [HIGH]   ZEEL   → SELL   — Awaiting fund manager approval
  [LOW]    AMBER  → HOLD   — Approved
```
