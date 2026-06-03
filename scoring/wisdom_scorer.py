"""
scoring/wisdom_scorer.py
Computes the WISDOM 10-point score from FundamentalSnapshot data.
This is purely numerical — LLMs are NEVER used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import THRESHOLDS, TRIGGERS, WisdomThresholds
from data.market_data import FundamentalSnapshot


@dataclass
class PrincipleScore:
    principle_id: int
    name: str
    score: float            # 0–10
    max_score: float = 10.0
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)  # (label, passed, detail)

    @property
    def pct(self) -> float:
        return self.score / self.max_score * 100


@dataclass
class WisdomScoreResult:
    ticker: str
    total_score: float          # 0–10 weighted average
    principles: List[PrincipleScore] = field(default_factory=list)
    signal: str = "WATCH"       # BUY | HOLD | SELL | WATCH
    trigger_reason: str = ""
    anti_panic_active: bool = False
    bias_flags: List[str] = field(default_factory=list)

    @property
    def buy_eligible(self) -> bool:
        return self.total_score >= TRIGGERS.buy_wisdom_score_min

    @property
    def hold_eligible(self) -> bool:
        return self.total_score >= TRIGGERS.hold_wisdom_score_min

    @property
    def sell_triggered(self) -> bool:
        return self.signal == "SELL"


class WisdomScorer:
    """
    Scores each stock on WISDOM's 5 principles using quantitative thresholds.
    Returns a WisdomScoreResult with per-principle breakdowns.
    """

    WEIGHTS = {1: 0.25, 2: 0.20, 3: 0.20, 4: 0.20, 5: 0.15}

    def __init__(self, thresholds: WisdomThresholds = THRESHOLDS):
        self.t = thresholds

    def score(
        self, snap: FundamentalSnapshot, macro_shock: bool = False
    ) -> WisdomScoreResult:
        p1 = self._principle1_business_quality(snap)
        p2 = self._principle2_skin_in_game(snap)
        p3 = self._principle3_reinvestment_runway(snap)
        p4 = self._principle4_structural_vs_cyclical(snap)
        p5 = self._principle5_balance_sheet(snap)

        principles = [p1, p2, p3, p4, p5]
        total = sum(
            p.score * self.WEIGHTS[p.principle_id] for p in principles
        ) / sum(self.WEIGHTS.values())

        result = WisdomScoreResult(
            ticker=snap.ticker,
            total_score=round(total, 2),
            principles=principles,
        )

        result.signal, result.trigger_reason, result.anti_panic_active = (
            self._determine_signal(result, snap, macro_shock)
        )
        result.bias_flags = self._detect_bias_flags(snap)
        return result

    # ── Principle 1: Business Quality ─────────────────────────────────────────

    def _principle1_business_quality(self, s: FundamentalSnapshot) -> PrincipleScore:
        checks = []
        score = 0.0

        # ROCE > 15% (3 pts)
        roce = s.roce_ttm or 0
        roce_5y = s.roce_5y_avg or 0
        if roce >= self.t.roce_min_pct:
            score += 3.0
            checks.append((f"ROCE {roce:.1f}% > {self.t.roce_min_pct}%", True, "Excellent"))
        elif roce >= self.t.roce_min_pct * 0.8:
            score += 1.5
            checks.append((f"ROCE {roce:.1f}% (below threshold, watch)", False, "Marginal"))
        else:
            checks.append((f"ROCE {roce:.1f}% < {self.t.roce_min_pct}%", False, "Fail"))

        # 5-year average ROCE (2 pts)
        if roce_5y >= self.t.roce_min_pct:
            score += 2.0
            checks.append((f"5Y Avg ROCE {roce_5y:.1f}%", True, "Durable"))
        else:
            checks.append((f"5Y Avg ROCE {roce_5y:.1f}% below threshold", False, "Cyclical concern"))

        # FCF Yield (3 pts)
        fcf = s.fcf_yield or 0
        if fcf >= self.t.fcf_yield_min_pct:
            score += 3.0
            checks.append((f"FCF Yield {fcf:.1f}% > {self.t.fcf_yield_min_pct}%", True, "Strong"))
        elif fcf >= self.t.fcf_yield_min_pct * 0.6:
            score += 1.5
            checks.append((f"FCF Yield {fcf:.1f}% (marginal)", False, "Marginal"))
        else:
            checks.append((f"FCF Yield {fcf:.1f}% (weak)", False, "Fail"))

        # Gross Margin (2 pts)
        gm = s.gross_margin or 0
        if gm >= 15.0:
            score += 2.0
            checks.append((f"Gross Margin {gm:.1f}%", True, "Healthy"))
        elif gm >= 8.0:
            score += 1.0
            checks.append((f"Gross Margin {gm:.1f}% (thin)", True, "Asset-intensive ok"))
        else:
            checks.append((f"Gross Margin {gm:.1f}% very thin", False, "Low quality"))

        return PrincipleScore(1, "Business Quality", round(min(score, 10), 2), checks=checks)

    # ── Principle 2: Skin in the Game ─────────────────────────────────────────

    def _principle2_skin_in_game(self, s: FundamentalSnapshot) -> PrincipleScore:
        checks = []
        score = 0.0

        ph = s.promoter_holding or 0
        if ph >= self.t.promoter_holding_min_pct:
            score += 5.0
            checks.append((f"Promoter {ph:.1f}% > {self.t.promoter_holding_min_pct}%", True, "Aligned"))
        elif ph >= 25.0:
            score += 2.5
            checks.append((f"Promoter {ph:.1f}% (moderate)", True, "Acceptable"))
        else:
            checks.append((f"Promoter {ph:.1f}% (very low)", False, "Governance risk"))

        pl = s.pledged_pct or 0
        if pl <= self.t.pledged_shares_max_pct:
            score += 5.0
            checks.append((f"Pledged {pl:.1f}% ≤ {self.t.pledged_shares_max_pct}%", True, "Clean"))
        else:
            checks.append((f"Pledged {pl:.1f}% (high risk)", False, "Governance concern"))

        return PrincipleScore(2, "Skin in the Game", round(min(score, 10), 2), checks=checks)

    # ── Principle 3: Reinvestment Runway ─────────────────────────────────────

    def _principle3_reinvestment_runway(self, s: FundamentalSnapshot) -> PrincipleScore:
        checks = []
        score = 0.0

        rr = s.retention_ratio or 0
        if rr >= self.t.retention_ratio_min_pct:
            score += 5.0
            checks.append((f"Retention {rr:.0f}% > {self.t.retention_ratio_min_pct}%", True, "Reinvesting"))
        elif rr >= 40.0:
            score += 2.5
            checks.append((f"Retention {rr:.0f}% (moderate)", True, "OK"))
        else:
            checks.append((f"Retention {rr:.0f}% (low — dividends over growth)", False, "Mature/Slow"))

        cr = s.capex_revenue_ratio or 0
        if 2.0 <= cr <= 12.0:
            score += 5.0
            checks.append((f"Capex/Rev {cr:.1f}% (active investment)", True, "Growing"))
        elif cr < 2.0:
            score += 2.0
            checks.append((f"Capex/Rev {cr:.1f}% (low investment)", False, "Shrinking footprint"))
        else:
            score += 3.0
            checks.append((f"Capex/Rev {cr:.1f}% (high, watch FCF)", True, "Capacity expansion"))

        return PrincipleScore(3, "Reinvestment Runway", round(min(score, 10), 2), checks=checks)

    # ── Principle 4: Structural vs Cyclical ──────────────────────────────────

    def _principle4_structural_vs_cyclical(self, s: FundamentalSnapshot) -> PrincipleScore:
        checks = []
        score = 0.0

        rb = s.revenue_beta or 0
        if rb <= self.t.revenue_beta_max:
            score += 5.0
            checks.append((f"Revenue Beta {rb:.2f} ≤ {self.t.revenue_beta_max}", True, "Stable/Structural"))
        elif rb <= 1.5:
            score += 2.5
            checks.append((f"Revenue Beta {rb:.2f} (moderate cyclicality)", False, "Semi-cyclical"))
        else:
            checks.append((f"Revenue Beta {rb:.2f} > 1.5 (HIGH cyclicality)", False, "Cyclical trap risk"))

        if not s.earnings_cyclicality:
            score += 5.0
            checks.append(("Earnings pattern: non-cyclical", True, "Structural compounder"))
        else:
            checks.append(("Earnings pattern: CYCLICAL", False, "Not a structural compounder"))

        return PrincipleScore(4, "Structural vs Cyclical", round(min(score, 10), 2), checks=checks)

    # ── Principle 5: Balance Sheet ────────────────────────────────────────────

    def _principle5_balance_sheet(self, s: FundamentalSnapshot) -> PrincipleScore:
        checks = []
        score = 0.0

        de = s.debt_equity or 0
        if de <= self.t.debt_equity_max:
            score += 5.0
            checks.append((f"D/E {de:.2f} ≤ {self.t.debt_equity_max}", True, "Conservative"))
        elif de <= 1.0:
            score += 2.5
            checks.append((f"D/E {de:.2f} (elevated, watch)", False, "Moderate leverage"))
        else:
            checks.append((f"D/E {de:.2f} > 1.0 (HIGH leverage)", False, "Balance sheet risk"))

        ic = s.interest_coverage or 0
        if ic >= self.t.interest_coverage_min:
            score += 5.0
            checks.append((f"Interest Coverage {ic:.1f}x ≥ {self.t.interest_coverage_min}x", True, "Comfortable"))
        elif ic >= 2.0:
            score += 2.5
            checks.append((f"Interest Coverage {ic:.1f}x (marginal)", False, "Watch carefully"))
        else:
            checks.append((f"Interest Coverage {ic:.1f}x (DANGER)", False, "Solvency risk"))

        return PrincipleScore(5, "Balance Sheet", round(min(score, 10), 2), checks=checks)

    # ── Signal Determination ──────────────────────────────────────────────────

    def _determine_signal(
        self,
        result: WisdomScoreResult,
        snap: FundamentalSnapshot,
        macro_shock: bool,
    ) -> Tuple[str, str, bool]:
        score = result.total_score
        roce  = snap.roce_ttm or 0
        de    = snap.debt_equity or 0
        ic    = snap.interest_coverage or 0

        anti_panic = False

        # Hard SELL triggers (fundamental deterioration)
        if roce > 0 and roce < TRIGGERS.sell_roce_below_pct:
            if de > 1.0 or ic < 2.5:
                return "SELL", f"ROCE {roce:.1f}% < {TRIGGERS.sell_roce_below_pct}% + balance sheet stress", False

        if score < 4.5:
            return "SELL", f"WISDOM score {score:.1f} < 4.5 — structural deterioration", False

        # Anti-panic HOLD lock (macro shock, not fundamental)
        if macro_shock and score >= TRIGGERS.hold_wisdom_score_min:
            anti_panic = True
            return "HOLD", "Anti-panic lock: macro shock only, fundamentals intact", True

        if score >= TRIGGERS.buy_wisdom_score_min:
            return "BUY", f"WISDOM {score:.1f} ≥ {TRIGGERS.buy_wisdom_score_min} (buy threshold)", False

        if score >= TRIGGERS.hold_wisdom_score_min:
            return "HOLD", f"WISDOM {score:.1f} — thesis intact, hold", False

        if score >= 5.5:
            return "WATCH", f"WISDOM {score:.1f} — below hold threshold, monitor closely", False

        return "SELL", f"WISDOM {score:.1f} — below minimum threshold", False

    def _detect_bias_flags(self, snap: FundamentalSnapshot) -> List[str]:
        flags = []
        if (snap.debt_equity or 0) > 0.8 and (snap.revenue_beta or 0) > 1.2:
            flags.append("cyclical_trap")
        if (snap.promoter_holding or 100) < 10:
            flags.append("governance_risk")
        return flags
