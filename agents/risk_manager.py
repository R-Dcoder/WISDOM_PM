"""
agents/risk_manager.py
Agent 3 — Risk Manager
Checks portfolio-level risk: concentration limits, cyclical exposure, governance flags.
Purely rule-based — no LLM needed for this agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import AUM_INR_CRORES, IDENTIFIED_BIASES, PORTFOLIO_STOCKS, THRESHOLDS
from data.market_data import FundamentalSnapshot
from scoring.wisdom_scorer import WisdomScoreResult


@dataclass
class RiskFlag:
    flag_id: str
    severity: str          # HIGH | MEDIUM | LOW
    category: str          # concentration | cyclical | governance | leverage | bias
    stock: Optional[str]
    description: str
    action: str            # what to do about it


@dataclass
class RiskAssessmentOutput:
    flags: List[RiskFlag] = field(default_factory=list)
    portfolio_risk_score: float = 0.0    # 0 (safe) – 10 (dangerous)
    cyclical_exposure_pct: float = 0.0
    max_concentration_pct: float = 0.0
    anti_panic_stocks: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def high_risk_flags(self) -> List[RiskFlag]:
        return [f for f in self.flags if f.severity == "HIGH"]

    @property
    def has_blockers(self) -> bool:
        return len(self.high_risk_flags) > 0


class RiskManagerAgent:
    """
    Agent 3: Risk Manager
    Pure rule-based — synthesises quant and qual outputs into portfolio risk flags.
    """

    # Max allowed single-stock allocation before concentration flag fires
    CONCENTRATION_LIMIT_PCT: float = 40.0
    CYCLICAL_EXPOSURE_LIMIT_PCT: float = 50.0

    def assess(
        self,
        scores: Dict[str, WisdomScoreResult],
        snapshots: Dict[str, FundamentalSnapshot],
        allocations: Optional[Dict[str, float]] = None,
        macro_shock: bool = False,
    ) -> RiskAssessmentOutput:

        alloc = allocations or {t: s["allocation_pct"] for t, s in PORTFOLIO_STOCKS.items()}
        out = RiskAssessmentOutput()

        # 1) Concentration risk
        out.max_concentration_pct = max(alloc.values(), default=0)
        if out.max_concentration_pct > self.CONCENTRATION_LIMIT_PCT:
            heavy = max(alloc, key=alloc.get)
            out.flags.append(RiskFlag(
                flag_id="concentration_high", severity="MEDIUM", category="concentration",
                stock=heavy,
                description=f"{heavy} at {alloc[heavy]}% of AUM exceeds {self.CONCENTRATION_LIMIT_PCT}% limit",
                action=f"Monitor {heavy} closely. Trim if WISDOM score drops below 8.",
            ))

        # 2) Cyclical exposure
        cyclical_tickers = [
            t for t, snap in snapshots.items()
            if snap.earnings_cyclicality or (snap.revenue_beta or 0) > 1.2
        ]
        out.cyclical_exposure_pct = sum(alloc.get(t, 0) for t in cyclical_tickers)
        if out.cyclical_exposure_pct > self.CYCLICAL_EXPOSURE_LIMIT_PCT:
            out.flags.append(RiskFlag(
                flag_id="cyclical_exposure", severity="HIGH", category="cyclical",
                stock=None,
                description=(
                    f"Cyclical stocks ({', '.join(cyclical_tickers)}) = "
                    f"{out.cyclical_exposure_pct:.0f}% of AUM — exceeds {self.CYCLICAL_EXPOSURE_LIMIT_PCT}% limit"
                ),
                action="Reduce cyclical exposure. Prioritise structural compounders.",
            ))

        # 3) Governance flags (per stock)
        for ticker, snap in snapshots.items():
            if (snap.promoter_holding or 100) < 10.0:
                out.flags.append(RiskFlag(
                    flag_id=f"governance_{ticker}", severity="HIGH", category="governance",
                    stock=ticker,
                    description=f"{ticker}: promoter holding {snap.promoter_holding:.2f}% — dangerously low",
                    action=f"Flag for immediate review. May trigger SELL if falls further.",
                ))
            if (snap.pledged_pct or 0) > THRESHOLDS.pledged_shares_max_pct:
                out.flags.append(RiskFlag(
                    flag_id=f"pledge_{ticker}", severity="MEDIUM", category="governance",
                    stock=ticker,
                    description=f"{ticker}: pledged shares {snap.pledged_pct:.1f}% > {THRESHOLDS.pledged_shares_max_pct}%",
                    action="Watch for forced selling if stock price falls.",
                ))

        # 4) Leverage flags
        for ticker, snap in snapshots.items():
            de = snap.debt_equity or 0
            if de > 1.0:
                out.flags.append(RiskFlag(
                    flag_id=f"leverage_{ticker}", severity="HIGH", category="leverage",
                    stock=ticker,
                    description=f"{ticker}: D/E = {de:.2f} > 1.0 — high leverage risk",
                    action="Reduce allocation. Monitor quarterly. Trigger SELL if IC < 2x.",
                ))
            elif de > THRESHOLDS.debt_equity_max:
                out.flags.append(RiskFlag(
                    flag_id=f"leverage_watch_{ticker}", severity="LOW", category="leverage",
                    stock=ticker,
                    description=f"{ticker}: D/E = {de:.2f} — above conservative threshold ({THRESHOLDS.debt_equity_max})",
                    action="Monitor trend. Alert if D/E crosses 0.8.",
                ))

        # 5) Anti-panic lock identification
        for ticker, score_result in scores.items():
            if score_result.anti_panic_active:
                out.anti_panic_stocks.append(ticker)
                out.flags.append(RiskFlag(
                    flag_id=f"antipanic_{ticker}", severity="LOW", category="bias",
                    stock=ticker,
                    description=f"{ticker}: anti-panic HOLD active — macro shock, fundamentals intact",
                    action="Do NOT execute any sell order until fundamentals change.",
                ))

        # 6) Investor bias flags
        for ticker, score_result in scores.items():
            for bias_id in score_result.bias_flags:
                bias = IDENTIFIED_BIASES.get(bias_id, {})
                if bias:
                    out.flags.append(RiskFlag(
                        flag_id=f"bias_{ticker}_{bias_id}", severity="MEDIUM", category="bias",
                        stock=ticker,
                        description=f"{ticker}: {bias['name']} — {bias['description'][:120]}",
                        action=bias.get("mitigation", "Apply WISDOM discipline"),
                    ))

        # 7) Portfolio risk score (0–10)
        score = 0.0
        for f in out.flags:
            score += {"HIGH": 2.5, "MEDIUM": 1.0, "LOW": 0.3}.get(f.severity, 0)
        out.portfolio_risk_score = round(min(score, 10.0), 2)

        out.summary = self._build_summary(out, alloc)
        return out

    def _build_summary(self, out: RiskAssessmentOutput, alloc: Dict[str, float]) -> str:
        total_aum = AUM_INR_CRORES
        lines = [
            f"Portfolio Risk Score: {out.portfolio_risk_score}/10",
            f"Cyclical Exposure: {out.cyclical_exposure_pct:.0f}% of AUM",
            f"Max Concentration: {out.max_concentration_pct:.0f}% (one stock)",
            f"High-Risk Flags: {len(out.high_risk_flags)}",
            f"Anti-Panic Locks: {out.anti_panic_stocks if out.anti_panic_stocks else 'None'}",
        ]
        return "\n".join(lines)
