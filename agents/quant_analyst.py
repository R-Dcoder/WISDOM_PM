"""
agents/quant_analyst.py
Agent 1 — Quant Analyst
Computes all numerical metrics and WISDOM scores.
Uses Tool-Calling so the LLM never performs arithmetic — it calls Python functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, PORTFOLIO_STOCKS, TRIGGERS
from data.market_data import FundamentalSnapshot, MarketDataFetcher
from scoring.wisdom_scorer import WisdomScorer, WisdomScoreResult

# ── Tool definitions (function-calling spec) ──────────────────────────────────

QUANT_TOOLS: List[Dict] = [
    {
        "name": "get_fundamental_snapshot",
        "description": (
            "Fetch the latest quantitative fundamental data for a stock ticker. "
            "Returns ROCE, FCF yield, D/E ratio, interest coverage, promoter holding, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g. AMBER, ZEEL)"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compute_wisdom_score",
        "description": (
            "Compute the WISDOM 5-principle score (0–10) for a stock using its fundamental data. "
            "Returns per-principle scores and the overall BUY/HOLD/SELL/WATCH signal. "
            "This function does all arithmetic — LLM must never compute scores manually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":      {"type": "string"},
                "macro_shock": {"type": "boolean", "description": "True if a macro/systemic shock is occurring"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "check_sell_triggers",
        "description": "Check whether any hard SELL triggers have been met for a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "check_buy_triggers",
        "description": "Check whether BUY conditions are satisfied (WISDOM score + valuation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
]


@dataclass
class QuantAnalystOutput:
    ticker: str
    snapshot: Optional[FundamentalSnapshot] = None
    score_result: Optional[WisdomScoreResult] = None
    sell_triggers: Dict[str, Any] = field(default_factory=dict)
    buy_triggers:  Dict[str, Any] = field(default_factory=dict)
    agent_summary: str = ""
    tool_calls_made: List[str] = field(default_factory=list)


class QuantAnalystAgent:
    """
    Agent 1: Quant Analyst
    Uses Anthropic tool-calling so the LLM orchestrates analysis
    without ever performing arithmetic itself.
    """

    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.scorer  = WisdomScorer()
        self._snapshots: Dict[str, FundamentalSnapshot] = {}
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    # ── Public entry point ────────────────────────────────────────────────────

    def analyse(self, ticker: str, macro_shock: bool = False) -> QuantAnalystOutput:
        """Run full quantitative analysis for one ticker."""
        output = QuantAnalystOutput(ticker=ticker)

        if not self.client:
            # Offline mode — run tools directly without LLM orchestration
            output.snapshot    = self._tool_get_snapshot(ticker)
            output.score_result = self._tool_compute_score(ticker, macro_shock)
            output.sell_triggers = self._tool_check_sell(ticker)
            output.buy_triggers  = self._tool_check_buy(ticker)
            output.agent_summary = self._offline_summary(output)
            return output

        # LLM-orchestrated tool-calling loop
        messages = [
            {
                "role": "user",
                "content": (
                    f"Perform a complete quantitative analysis for {ticker}. "
                    f"macro_shock={macro_shock}. "
                    "Use the available tools to: "
                    "1) fetch fundamental data, "
                    "2) compute WISDOM score, "
                    "3) check sell triggers, "
                    "4) check buy triggers. "
                    "Then provide a concise quant summary (3–5 bullet points). "
                    "IMPORTANT: never compute any numbers yourself — always call the tools."
                ),
            }
        ]

        while True:
            resp = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=1500,
                tools=QUANT_TOOLS,
                messages=messages,
            )

            # Collect any text the model produced
            text_parts = [b.text for b in resp.content if b.type == "text"]
            if text_parts:
                output.agent_summary = "\n".join(text_parts)

            # If no tool calls — we're done
            if resp.stop_reason == "end_turn":
                break

            # Process tool calls
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue

                output.tool_calls_made.append(block.name)
                result_data = self._dispatch_tool(block.name, block.input, ticker, macro_shock)

                # Cache structured outputs
                if block.name == "get_fundamental_snapshot":
                    output.snapshot = self._snapshots.get(ticker)
                elif block.name == "compute_wisdom_score":
                    output.score_result = result_data.get("_obj")
                elif block.name == "check_sell_triggers":
                    output.sell_triggers = result_data
                elif block.name == "check_buy_triggers":
                    output.buy_triggers  = result_data

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps({k: v for k, v in result_data.items() if k != "_obj"}),
                })

            # Feed tool results back into conversation
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user",      "content": tool_results})

        return output

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, inputs: dict, ticker: str, macro_shock: bool) -> dict:
        t = inputs.get("ticker", ticker)
        if name == "get_fundamental_snapshot":
            return self._tool_get_snapshot_dict(t)
        if name == "compute_wisdom_score":
            return self._tool_compute_score_dict(t, inputs.get("macro_shock", macro_shock))
        if name == "check_sell_triggers":
            return self._tool_check_sell(t)
        if name == "check_buy_triggers":
            return self._tool_check_buy(t)
        return {"error": f"Unknown tool: {name}"}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_get_snapshot(self, ticker: str) -> FundamentalSnapshot:
        if ticker not in self._snapshots:
            cfg = PORTFOLIO_STOCKS.get(ticker, {})
            snap = self.fetcher.fetch(ticker, cfg.get("yf_ticker", ticker + ".NS"), cfg.get("name", ticker))
            self._snapshots[ticker] = snap
        return self._snapshots[ticker]

    def _tool_get_snapshot_dict(self, ticker: str) -> dict:
        s = self._tool_get_snapshot(ticker)
        return {
            "ticker": s.ticker, "name": s.name,
            "roce_ttm": s.roce_ttm, "roce_5y_avg": s.roce_5y_avg,
            "fcf_yield": s.fcf_yield, "gross_margin": s.gross_margin,
            "promoter_holding": s.promoter_holding, "pledged_pct": s.pledged_pct,
            "retention_ratio": s.retention_ratio, "capex_revenue_ratio": s.capex_revenue_ratio,
            "revenue_beta": s.revenue_beta, "earnings_cyclicality": s.earnings_cyclicality,
            "debt_equity": s.debt_equity, "interest_coverage": s.interest_coverage,
            "pe_ratio": s.pe_ratio, "peg_ratio": s.peg_ratio,
            "price": s.price, "market_cap_cr": s.market_cap_cr,
            "fetch_error": s.fetch_error,
        }

    def _tool_compute_score(self, ticker: str, macro_shock: bool = False) -> WisdomScoreResult:
        snap = self._tool_get_snapshot(ticker)
        return self.scorer.score(snap, macro_shock=macro_shock)

    def _tool_compute_score_dict(self, ticker: str, macro_shock: bool = False) -> dict:
        r = self._tool_compute_score(ticker, macro_shock)
        return {
            "_obj": r,
            "ticker": r.ticker, "total_score": r.total_score, "signal": r.signal,
            "trigger_reason": r.trigger_reason, "anti_panic_active": r.anti_panic_active,
            "bias_flags": r.bias_flags,
            "principles": [
                {"id": p.principle_id, "name": p.name, "score": p.score}
                for p in r.principles
            ],
        }

    def _tool_check_sell(self, ticker: str) -> dict:
        s = self._tool_get_snapshot(ticker)
        triggers = {}
        if (s.roce_ttm or 0) < TRIGGERS.sell_roce_below_pct and (s.roce_ttm or 0) > 0:
            triggers["roce_below_threshold"] = {
                "fired": True,
                "detail": f"ROCE {s.roce_ttm:.1f}% < {TRIGGERS.sell_roce_below_pct}%",
            }
        if (s.debt_equity or 0) > 1.0:
            triggers["leverage_excessive"] = {
                "fired": True,
                "detail": f"D/E {s.debt_equity:.2f} > 1.0",
            }
        if (s.interest_coverage or 10) < 2.5:
            triggers["interest_coverage_low"] = {
                "fired": True,
                "detail": f"IC {s.interest_coverage:.1f}x < 2.5x",
            }
        if (s.promoter_holding or 100) < 10.0:
            triggers["promoter_very_low"] = {
                "fired": True,
                "detail": f"Promoter holding {s.promoter_holding:.2f}% — governance risk",
            }
        triggers["any_fired"] = any(v.get("fired") for v in triggers.values())
        return triggers

    def _tool_check_buy(self, ticker: str) -> dict:
        s = self._tool_get_snapshot(ticker)
        r = self._tool_compute_score(ticker)
        return {
            "wisdom_score_ok":  r.total_score >= TRIGGERS.buy_wisdom_score_min,
            "peg_ok":           (s.peg_ratio or 99) <= TRIGGERS.buy_peg_max,
            "wisdom_score":     r.total_score,
            "peg_ratio":        s.peg_ratio,
            "buy_eligible":     r.total_score >= TRIGGERS.buy_wisdom_score_min and (s.peg_ratio or 99) <= TRIGGERS.buy_peg_max,
        }

    def _offline_summary(self, output: QuantAnalystOutput) -> str:
        r = output.score_result
        if not r:
            return "No score computed."
        lines = [f"WISDOM Score: {r.total_score}/10 → {r.signal}"]
        for p in r.principles:
            lines.append(f"  • P{p.principle_id} {p.name}: {p.score}/10")
        if r.trigger_reason:
            lines.append(f"Trigger: {r.trigger_reason}")
        return "\n".join(lines)
