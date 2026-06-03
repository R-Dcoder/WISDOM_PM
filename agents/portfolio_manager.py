"""
agents/portfolio_manager.py
Agent 4 — Portfolio Manager (Orchestrator)
Synthesises outputs from Agents 1–3 and produces a Trade Recommendation Memo.
Uses the LLM for final narrative synthesis. Human sign-off is required before execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import anthropic

from config import ANTHROPIC_API_KEY, AUM_INR_CRORES, LLM_MODEL, PORTFOLIO_STOCKS
from agents.quant_analyst import QuantAnalystOutput
from agents.qual_researcher import QualResearchOutput
from agents.risk_manager import RiskAssessmentOutput, RiskFlag
from scoring.wisdom_scorer import WisdomScoreResult


@dataclass
class TradeRecommendationMemo:
    """Human-in-the-Loop memo — must be signed off before any order is placed."""

    ticker: str
    stock_name: str
    recommendation: str              # BUY | HOLD | SELL | WATCH
    urgency: str                     # HIGH | MEDIUM | LOW
    wisdom_score: float
    generated_at: str = ""

    # Structured content
    agent_syntheses: List[str] = field(default_factory=list)
    rationale: str = ""
    instruction: str = ""
    risk_note: str = ""
    bias_override_applied: bool = False
    anti_panic_active: bool = False

    # HITL state
    approved: Optional[bool] = None   # None = pending, True = approved, False = rejected
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "stock_name": self.stock_name,
            "recommendation": self.recommendation, "urgency": self.urgency,
            "wisdom_score": self.wisdom_score, "generated_at": self.generated_at,
            "rationale": self.rationale, "instruction": self.instruction,
            "risk_note": self.risk_note,
            "bias_override": self.bias_override_applied,
            "anti_panic": self.anti_panic_active,
            "hitl_status": "PENDING" if self.approved is None else ("APPROVED" if self.approved else "REJECTED"),
        }

    def approve(self, by: str = "Fund Manager") -> None:
        self.approved    = True
        self.approved_by = by
        self.approved_at = datetime.now().isoformat()

    def reject(self, by: str = "Fund Manager") -> None:
        self.approved    = False
        self.approved_by = by
        self.approved_at = datetime.now().isoformat()


@dataclass
class PMOrchestrationOutput:
    memos: List[TradeRecommendationMemo] = field(default_factory=list)
    portfolio_summary: str = ""
    step1_profile_summary: str = ""
    step2_decision_summary: str = ""
    step3_architecture_note: str = ""

    @property
    def pending_memos(self) -> List[TradeRecommendationMemo]:
        return [m for m in self.memos if m.approved is None]

    @property
    def sell_memos(self) -> List[TradeRecommendationMemo]:
        return [m for m in self.memos if m.recommendation == "SELL"]


class PortfolioManagerAgent:
    """
    Agent 4: Portfolio Manager (WISDOM-PM Orchestrator)
    Runs all 3 sub-agents in sequence and synthesises a Trade Recommendation Memo
    for every stock in the portfolio.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    def generate_memos(
        self,
        quant_outputs: Dict[str, QuantAnalystOutput],
        qual_outputs:  Dict[str, QualResearchOutput],
        risk_output:   RiskAssessmentOutput,
    ) -> PMOrchestrationOutput:

        result = PMOrchestrationOutput()

        for ticker in PORTFOLIO_STOCKS:
            qn  = quant_outputs.get(ticker)
            ql  = qual_outputs.get(ticker)
            risk_flags = [f for f in risk_output.flags if f.stock == ticker]

            memo = self._build_memo(ticker, qn, ql, risk_flags, risk_output)
            result.memos.append(memo)

        result.portfolio_summary  = self._portfolio_summary(quant_outputs, risk_output)
        result.step1_profile_summary = self._step1_summary()
        result.step2_decision_summary = self._step2_summary(quant_outputs)
        result.step3_architecture_note = self._step3_note()

        return result

    # ── Memo generation ───────────────────────────────────────────────────────

    def _build_memo(
        self,
        ticker: str,
        qn: Optional[QuantAnalystOutput],
        ql: Optional[QualResearchOutput],
        risk_flags: List[RiskFlag],
        risk_out: RiskAssessmentOutput,
    ) -> TradeRecommendationMemo:

        score_result = qn.score_result if qn else None
        snap         = qn.snapshot     if qn else None
        signal       = score_result.signal if score_result else "WATCH"
        wisdom       = score_result.total_score if score_result else 0.0
        stock_cfg    = PORTFOLIO_STOCKS.get(ticker, {})

        memo = TradeRecommendationMemo(
            ticker=ticker,
            stock_name=stock_cfg.get("name", ticker),
            recommendation=signal,
            wisdom_score=wisdom,
            urgency=self._urgency(signal, wisdom),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            anti_panic_active=score_result.anti_panic_active if score_result else False,
        )

        # Agent syntheses
        if qn and score_result:
            memo.agent_syntheses.append(
                f"[Agent 1 — Quant] WISDOM {wisdom:.1f}/10. "
                + " | ".join(
                    f"P{p.principle_id}: {p.score:.1f}"
                    for p in score_result.principles
                )
            )
        if ql:
            memo.agent_syntheses.append(
                f"[Agent 2 — Qual] Sentiment: {ql.analyst_sentiment.upper()}. "
                f"Thesis intact: {ql.thesis_intact}. "
                + (f"Break evidence: {ql.thesis_break_evidence[0][:80]}..." if ql.thesis_break_evidence else "")
            )
        if risk_flags:
            flag_str = "; ".join(f"{f.severity}:{f.flag_id}" for f in risk_flags)
            memo.agent_syntheses.append(f"[Agent 3 — Risk] Flags: {flag_str}")
        else:
            memo.agent_syntheses.append("[Agent 3 — Risk] No portfolio-level flags for this stock.")

        # Use LLM for narrative if available, else use rule-based
        if self.client:
            memo.rationale, memo.instruction, memo.risk_note = self._llm_narrative(
                ticker, memo, qn, ql, risk_flags
            )
        else:
            memo.rationale, memo.instruction, memo.risk_note = self._rule_based_narrative(
                ticker, signal, wisdom, snap, ql, risk_flags, score_result
            )

        # Bias override flag
        memo.bias_override_applied = bool(
            ql and ql.bias_flags or
            (score_result and score_result.bias_flags)
        )

        return memo

    def _llm_narrative(self, ticker, memo, qn, ql, risk_flags) -> tuple[str, str, str]:
        """Ask the LLM to write the 3 memo sections from structured inputs."""
        context = {
            "ticker": ticker,
            "recommendation": memo.recommendation,
            "wisdom_score": memo.wisdom_score,
            "agent_syntheses": memo.agent_syntheses,
            "anti_panic": memo.anti_panic_active,
            "risk_flags": [f.description for f in risk_flags],
            "qual_sentiment": ql.analyst_sentiment if ql else "unknown",
            "thesis_intact": ql.thesis_intact if ql else True,
        }
        prompt = (
            f"You are the Portfolio Manager for WISDOM-PM, a disciplined Indian equity fund. "
            f"Based on the following analysis inputs, write a Trade Recommendation Memo for {ticker}.\n\n"
            f"Inputs: {json.dumps(context, indent=2)}\n\n"
            "Write exactly three sections:\n"
            "RATIONALE: (1–2 sentences explaining WHY this recommendation)\n"
            "INSTRUCTION: (precise, actionable order instruction for the fund manager)\n"
            "RISK NOTE: (1–2 sentences on the key risk or caveat)\n\n"
            "Be concise and professional. No preamble."
        )
        try:
            resp = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            rationale = instruction = risk_note = ""
            for line in text.splitlines():
                if line.startswith("RATIONALE:"):
                    rationale = line[len("RATIONALE:"):].strip()
                elif line.startswith("INSTRUCTION:"):
                    instruction = line[len("INSTRUCTION:"):].strip()
                elif line.startswith("RISK NOTE:"):
                    risk_note = line[len("RISK NOTE:"):].strip()
            return rationale or text[:200], instruction or "See full memo.", risk_note or ""
        except Exception as e:
            return self._rule_based_narrative(
                ticker, memo.recommendation, memo.wisdom_score,
                qn.snapshot if qn else None, None, risk_flags,
                qn.score_result if qn else None
            )

    def _rule_based_narrative(self, ticker, signal, wisdom, snap, ql, risk_flags, score_result) -> tuple[str, str, str]:
        alloc = PORTFOLIO_STOCKS.get(ticker, {}).get("allocation_pct", 0)
        aum_amt = AUM_INR_CRORES * alloc / 100

        if signal == "SELL":
            rationale  = (f"WISDOM score {wisdom:.1f}/10 indicates structural deterioration. "
                          f"{'Thesis break confirmed in analyst corpus. ' if ql and not ql.thesis_intact else ''}"
                          "Continuing to hold risks further capital destruction.")
            instruction = (f"Exit 100% of {ticker} position (₹{aum_amt:.1f} Cr) at market on next trading session. "
                           "Proceeds to cash pending redeployment in higher-conviction ideas.")
            risk_note   = "Sunk-cost bias override applied — do not average down."

        elif signal == "BUY":
            rationale  = (f"WISDOM score {wisdom:.1f}/10 — all 5 principles satisfied. "
                          f"{'Analyst sentiment bullish. ' if ql and ql.analyst_sentiment == 'bullish' else ''}"
                          "Structural thesis intact with reinvestment runway visible.")
            instruction = (f"Add to {ticker} position. Increase allocation to {min(alloc + 5, 40)}% of AUM. "
                           "Buy in tranches over 5 trading days to average entry.")
            risk_note   = "Do not chase momentum. Entry only at or below current market price."

        elif signal == "HOLD":
            anti = score_result and score_result.anti_panic_active
            rationale  = ("Anti-panic mechanism active — macro shock detected but fundamentals intact. "
                          if anti else
                          f"WISDOM score {wisdom:.1f}/10 — thesis intact. No action required.")
            instruction = f"Hold existing {ticker} position ({alloc}% of AUM = ₹{aum_amt:.1f} Cr). No trade."
            risk_note   = ("Portfolio frozen until macro shock resolves. Monitor ROCE quarterly."
                           if anti else "Review on next quarterly earnings.")
        else:  # WATCH
            rationale  = (f"WISDOM score {wisdom:.1f}/10 — below HOLD threshold. "
                          "Metrics trending in wrong direction but no hard sell trigger yet.")
            instruction = f"Reduce {ticker} allocation from {alloc}% to {max(alloc - 5, 10)}% of AUM on strength."
            risk_note   = f"Set alert: SELL if ROCE drops below 10% or D/E crosses 1.5."

        return rationale, instruction, risk_note

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _urgency(self, signal: str, wisdom: float) -> str:
        if signal == "SELL" or wisdom < 4.5:
            return "HIGH"
        if signal == "WATCH" or wisdom < 7.0:
            return "MEDIUM"
        return "LOW"

    def _portfolio_summary(self, quant: Dict, risk: RiskAssessmentOutput) -> str:
        scores = {t: q.score_result.total_score for t, q in quant.items() if q.score_result}
        avg    = sum(scores.values()) / len(scores) if scores else 0
        lines  = [
            f"AUM: ₹{AUM_INR_CRORES} Crores  |  Avg WISDOM Score: {avg:.1f}/10",
            f"Portfolio Risk Score: {risk.portfolio_risk_score}/10",
            f"Cyclical Exposure: {risk.cyclical_exposure_pct:.0f}%",
            "Signals: " + "  ".join(
                f"{t}={q.score_result.signal}" for t, q in quant.items() if q.score_result
            ),
        ]
        return "\n".join(lines)

    def _step1_summary(self) -> str:
        return (
            "Step 1 — Investor Profiling Complete.\n"
            "Strengths  : Exceptional patience on structural themes (AMBER 8-year hold).\n"
            "Bias 1     : Macro Panic — sold WELSPUN at ₹21 (–70%) during Covid.\n"
            "Bias 2     : Sunk Cost  — averaged down ZEEL through structural decline.\n"
            "Bias 3     : Cyclical Trap — misread DBL order-book peak as structural growth."
        )

    def _step2_summary(self, quant: Dict) -> str:
        lines = ["Step 2 — WISDOM Decision Matrix Results:"]
        for ticker, q in quant.items():
            if q.score_result:
                lines.append(
                    f"  {ticker:10s}: {q.score_result.total_score:.1f}/10 → {q.score_result.signal}"
                    + (f"  [anti-panic]" if q.score_result.anti_panic_active else "")
                )
        return "\n".join(lines)

    def _step3_note(self) -> str:
        return (
            "Step 3 — Architecture:\n"
            "  Data     : Yahoo Finance / Screener.in → ETL → PostgreSQL + S3\n"
            "  RAG      : Analyst PDFs + Concalls → ChromaDB vector store\n"
            "  Agents   : LangGraph (Quant → Qual → Risk → PM)\n"
            "  Guardrail: LLM never computes numbers — Tool-Calling only\n"
            "  Privacy  : Trade history never leaves encrypted on-premise DB\n"
            "  HITL     : All memos require digital sign-off before execution"
        )
