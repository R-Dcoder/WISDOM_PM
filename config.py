"""
WISDOM-PM Configuration
Central config for thresholds, tickers, and system parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


# ─── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-sonnet-4-20250514"


# ─── Portfolio ────────────────────────────────────────────────────────────────
AUM_INR_CRORES = 10.0

PORTFOLIO_STOCKS = {
    "AMBER":   {"name": "Amber Enterprises India Ltd",  "yf_ticker": "AMBER.NS",   "sector": "EMS / AC Manufacturing",    "allocation_pct": 35},
    "WELSPUN": {"name": "Welspun Living Ltd",           "yf_ticker": "WELSPUNLIV.NS","sector": "Textiles / FTA Beneficiary","allocation_pct": 25},
    "ZEEL":    {"name": "Zee Entertainment Enterprises","yf_ticker": "ZEEL.NS",    "sector": "Media / Linear TV",          "allocation_pct": 20},
    "DBL":     {"name": "Dilip Buildcon Ltd",           "yf_ticker": "DBL.NS",     "sector": "EPC / Infrastructure",       "allocation_pct": 20},
}


# ─── WISDOM Score Thresholds ───────────────────────────────────────────────────
@dataclass
class WisdomThresholds:
    """Quantitative thresholds for all 5 WISDOM principles."""

    # Principle 1 — Business Quality
    roce_min_pct: float = 15.0          # ROCE must exceed this
    roce_years_min: int = 5             # In at least N of last 10 years
    fcf_yield_min_pct: float = 5.0      # Free Cash Flow yield minimum

    # Principle 2 — Skin in the Game
    promoter_holding_min_pct: float = 40.0
    pledged_shares_max_pct: float = 5.0

    # Principle 3 — Reinvestment Runway
    retention_ratio_min_pct: float = 60.0   # Earnings retained (1 – payout ratio)
    capex_revenue_trending: bool = True      # Capex/Revenue should be rising

    # Principle 4 — Structural vs Cyclical
    revenue_beta_max: float = 1.2           # Revenue volatility beta
    cyclical_flag: bool = False             # True if earnings are cyclical

    # Principle 5 — Balance Sheet
    debt_equity_max: float = 0.5
    interest_coverage_min: float = 4.0


# ─── Signal Triggers ──────────────────────────────────────────────────────────
@dataclass
class SignalTriggers:
    """BUY / HOLD / SELL threshold definitions."""

    # BUY
    buy_wisdom_score_min: float = 8.0
    buy_peg_max: float = 1.5
    buy_dcf_upside_min_pct: float = 25.0

    # HOLD (Anti-panic mechanism)
    hold_wisdom_score_min: float = 7.0
    hold_price_drop_macro_pct: float = 20.0   # If drop > 20% due to macro → freeze

    # SELL
    sell_roce_below_pct: float = 12.0
    sell_roce_consecutive_quarters: int = 2    # Consecutive quarters below threshold
    sell_governance_flag: bool = True          # Promoter pledge spike or RPT


THRESHOLDS = WisdomThresholds()
TRIGGERS = SignalTriggers()


# ─── Investor Bias Profiles ───────────────────────────────────────────────────
IDENTIFIED_BIASES = {
    "macro_panic": {
        "name": "Recency / Macro Panic Bias",
        "description": (
            "Prone to panic-selling high-quality assets during systemic shocks "
            "(e.g., Welspun Mar 2020 at ₹21, a -70% loss). Business durability "
            "ignored during liquidity crises."
        ),
        "affected_stocks": ["WELSPUN"],
        "mitigation": "Anti-panic HOLD lock: WISDOM Score > 7 + macro-only drop → portfolio freeze",
    },
    "sunk_cost": {
        "name": "Sunk Cost Fallacy",
        "description": (
            "Reluctance to exit structurally impaired businesses. Repeated averaging "
            "down in Zee Entertainment despite structural linear TV disruption."
        ),
        "affected_stocks": ["ZEEL"],
        "mitigation": "Thesis-break SELL trigger: ROCE < 12% for 2 consecutive quarters auto-generates SELL memo",
    },
    "cyclical_trap": {
        "name": "Cyclical Trap",
        "description": (
            "Misinterpreting cyclical order-book peaks as structural durability. "
            "DBL's EPC peaks treated as compounding signal."
        ),
        "affected_stocks": ["DBL"],
        "mitigation": "Revenue Beta check > 1.2 flags cyclical pattern and downgrades Principle 4 score",
    },
}


# ─── Historical Trade Snapshots (Step 1 data) ─────────────────────────────────
HISTORICAL_TRADES = [
    {
        "ticker": "AMBER", "action": "BUY",  "year": 2017, "price": 835,
        "qty": 400, "thesis": "India EMS/AC manufacturing structural megatrend, PLI beneficiary",
    },
    {
        "ticker": "AMBER", "action": "HOLD", "year": 2021, "price": 3200,
        "qty": 400, "thesis": "Structural theme intact, PCBA margins expanding",
    },
    {
        "ticker": "WELSPUN", "action": "BUY",  "year": 2018, "price": 72,
        "qty": 1500, "thesis": "Home textile export leader, balance sheet improving",
    },
    {
        "ticker": "WELSPUN", "action": "SELL", "year": 2020, "price": 21,
        "qty": 1500, "thesis": "PANIC SELL — Covid liquidity crisis (bias: macro panic)",
        "bias": "macro_panic",
    },
    {
        "ticker": "WELSPUN", "action": "BUY",  "year": 2020, "price": 38,
        "qty": 1500, "thesis": "Re-entry — India-UK FTA tailwind, balance sheet recovery",
    },
    {
        "ticker": "ZEEL", "action": "BUY",  "year": 2019, "price": 340,
        "qty": 800, "thesis": "Large-cap media, dividend yield play",
    },
    {
        "ticker": "ZEEL", "action": "BUY",  "year": 2020, "price": 175,
        "qty": 500, "thesis": "Averaging down (bias: sunk cost fallacy)",
        "bias": "sunk_cost",
    },
    {
        "ticker": "ZEEL", "action": "BUY",  "year": 2022, "price": 220,
        "qty": 300, "thesis": "Sony merger optimism (bias: sunk cost fallacy)",
        "bias": "sunk_cost",
    },
    {
        "ticker": "DBL",  "action": "BUY",  "year": 2021, "price": 580,
        "qty": 600, "thesis": "Infrastructure capex supercycle, record order book",
    },
    {
        "ticker": "DBL",  "action": "HOLD", "year": 2022, "price": 370,
        "qty": 600, "thesis": "Holding through cycle turn (bias: cyclical trap)",
        "bias": "cyclical_trap",
    },
]


# ─── Sample RAG Documents (analyst notes + concall excerpts) ──────────────────
SAMPLE_ANALYST_DOCS = [
    {
        "id": "amber_kotak_2024",
        "ticker": "AMBER",
        "source": "Kotak Institutional Equities",
        "date": "2024-11",
        "text": (
            "Amber Enterprises remains our top pick in the EMS space. The company's "
            "PCBA business is scaling rapidly, with margins improving from 4.2% to 6.8% "
            "YoY. PLI scheme disbursals are accelerating, and management has guided "
            "for 35% revenue CAGR in the PCBA segment over FY25-27. Pricing power "
            "on AC components remains intact despite competitive pressures from Dixon. "
            "ROCE has improved to 22.4% on a TTM basis, well above our 15% threshold. "
            "Maintain BUY with target price ₹6,800."
        ),
    },
    {
        "id": "welspun_anand_rathi_2024",
        "ticker": "WELSPUN",
        "source": "Anand Rathi Securities",
        "date": "2024-10",
        "text": (
            "Welspun Living is well-positioned to capture India-UK FTA tailwinds. "
            "The company's net debt/EBITDA has declined to 0.8x from 2.1x in FY20. "
            "Home textile exports are growing at 18% YoY driven by the UK FTA and "
            "China+1 strategy. Management tone on capital allocation has turned "
            "significantly more conservative, with no large debt-funded acquisitions "
            "planned. Balance sheet quality has improved materially. Target: ₹210."
        ),
    },
    {
        "id": "zeel_jm_financial_2024",
        "ticker": "ZEEL",
        "source": "JM Financial",
        "date": "2024-09",
        "text": (
            "Zee Entertainment's structural challenges are deepening. Linear TV ad "
            "revenue declined 12% YoY as OTT platforms accelerate cord-cutting. "
            "Zee5 losses widened to ₹450 Cr in FY24 with no clear path to profitability. "
            "ROCE has dropped to 8.3% — the third consecutive quarter below the 12% "
            "threshold. Adverse operating leverage from fixed content costs is compressing "
            "margins. The Sony merger collapse removed the last significant re-rating "
            "catalyst. SELL. Target ₹90. No visible catalyst for recovery."
        ),
    },
    {
        "id": "dbl_kotak_2024",
        "ticker": "DBL",
        "source": "Kotak Institutional Equities",
        "date": "2024-08",
        "text": (
            "Dilip Buildcon's order book peak in FY22 was a cyclical phenomenon, not "
            "structural. D/E ratio has risen to 1.12x as working capital cycles elongate. "
            "Revenue Beta over the last 5 years exceeds 1.5, confirming high cyclicality. "
            "ROCE has declined to 11.8%, approaching our 10% SELL threshold. Infrastructure "
            "sector CAPEX from government has slowed in election year. EPC businesses "
            "typically trade at deep discounts to capital-light compounders. REDUCE."
        ),
    },
    {
        "id": "amber_concall_q2fy25",
        "ticker": "AMBER",
        "source": "Q2 FY25 Earnings Concall Transcript",
        "date": "2024-11",
        "text": (
            "Management commentary on capital allocation: 'We are committed to a "
            "debt-free balance sheet. Our PCBA capacity expansion is 100% internally "
            "funded through operating cash flows. We see no reason to raise equity at "
            "current valuations.' On competition: 'Our technical moat in room AC "
            "components is defensible — we have 10+ years of manufacturing know-how "
            "that cannot be replicated quickly.' On PLI: 'FY25 disbursals are on track. "
            "We expect ₹180 Cr of PLI inflows in H2 FY25.'"
        ),
    },
    {
        "id": "zeel_concall_q2fy25",
        "ticker": "ZEEL",
        "source": "Q2 FY25 Earnings Concall Transcript",
        "date": "2024-10",
        "text": (
            "Management on linear TV: 'We acknowledge the headwinds in the linear TV "
            "advertising market. However, we believe our content library provides "
            "long-term value.' On Zee5: 'Zee5 losses are expected to continue through "
            "FY26 as we invest in content.' On promoter stake: 'We are exploring "
            "strategic options for monetisation.' Analyst note: Management tone "
            "significantly more cautious vs Q2 FY24. No specific guidance on when "
            "Zee5 reaches EBITDA break-even. Promoter stake at 3.99% — governance risk."
        ),
    },
]
