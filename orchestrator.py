"""
WISDOM-PM Full Pipeline Orchestrator
Runs the complete 3-step framework:
  Step 1 → Historical Trade Analysis & Investor Profiling
  Step 2 → WISDOM Decision Matrix (4 agents, parallel scoring)
  Step 3 → Architecture checks + Trade Memo generation

Now integrates real trade data from Excel ledger.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Optional

from config import PORTFOLIO_STOCKS, get_trade_ledger
from agents.quant_analyst import QuantAnalystAgent, QuantAnalystOutput
from agents.qual_researcher import QualResearcherAgent, QualResearchOutput
from agents.risk_manager import RiskManagerAgent, RiskAssessmentOutput
from agents.portfolio_manager import PortfolioManagerAgent, PMOrchestrationOutput
from data.market_data import FundamentalSnapshot
from rag.vector_store import WisdomVectorStore
from scoring.wisdom_scorer import WisdomScoreResult


class WisdomPMOrchestrator:
    """
    Top-level orchestrator that wires together all four agents
    and runs the 3-step WISDOM-PM framework.
    """

    def __init__(self, macro_shock: bool = False, data_dir: Optional[str] = None):
        self.macro_shock = macro_shock
        self.data_dir = data_dir
        
        # Load trade ledger
        self.trade_ledger = get_trade_ledger()
        
        self.vs = WisdomVectorStore()
        self.quant_agent = QuantAnalystAgent()
        self.qual_agent = QualResearcherAgent(vector_store=self.vs)
        self.risk_agent = RiskManagerAgent()
        self.pm_agent = PortfolioManagerAgent()

        # Shared state filled during run
        self.quant_outputs: Dict[str, QuantAnalystOutput] = {}
        self.qual_outputs: Dict[str, QualResearchOutput] = {}
        self.snapshots: Dict[str, FundamentalSnapshot] = {}
        self.scores: Dict[str, WisdomScoreResult] = {}
        self.risk_output: Optional[RiskAssessmentOutput] = None
        self.pm_output: Optional[PMOrchestrationOutput] = None
        self.run_timestamp: str = ""

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> PMOrchestrationOutput:
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── STEP 1: Run Quant Agent for every stock ───────────────────────────
        for ticker in PORTFOLIO_STOCKS:
            qn = self.quant_agent.analyse(ticker, macro_shock=self.macro_shock)
            self.quant_outputs[ticker] = qn
            if qn.snapshot:
                self.snapshots[ticker] = qn.snapshot
            if qn.score_result:
                self.scores[ticker] = qn.score_result

        # ── STEP 2: Run Qual Agent for every stock ────────────────────────────
        for ticker in PORTFOLIO_STOCKS:
            signal = self.scores[ticker].signal if ticker in self.scores else "WATCH"
            ql = self.qual_agent.research(ticker, quant_signal=signal)
            self.qual_outputs[ticker] = ql

        # ── STEP 3: Risk Manager + Portfolio Manager ──────────────────────────
        self.risk_output = self.risk_agent.assess(
            scores=self.scores,
            snapshots=self.snapshots,
            macro_shock=self.macro_shock,
        )

        self.pm_output = self.pm_agent.generate_memos(
            quant_outputs=self.quant_outputs,
            qual_outputs=self.qual_outputs,
            risk_output=self.risk_output,
        )

        return self.pm_output

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_json(self) -> str:
        if not self.pm_output:
            return "{}"
        return json.dumps({
            "run_at": self.run_timestamp,
            "portfolio_summary": self.pm_output.portfolio_summary,
            "step1": self.pm_output.step1_profile_summary,
            "step2": self.pm_output.step2_decision_summary,
            "step3": self.pm_output.step3_architecture_note,
            "memos": [m.to_dict() for m in self.pm_output.memos],
            "risk": {
                "score": self.risk_output.portfolio_risk_score if self.risk_output else 0,
                "flags": [
                    {"id": f.flag_id, "severity": f.severity, "desc": f.description}
                    for f in (self.risk_output.flags if self.risk_output else [])
                ],
            },
            "trade_ledger": {
                "total_trades": len(self.trade_ledger.trades) if self.trade_ledger else 0,
                "open_positions": len([p for p in self.trade_ledger.positions.values() if p.qty > 0]) if self.trade_ledger else 0,
            } if self.trade_ledger else {},
        }, indent=2)

    def approve_memo(self, ticker: str, by: str = "Fund Manager") -> bool:
        if not self.pm_output:
            return False
        for memo in self.pm_output.memos:
            if memo.ticker == ticker:
                memo.approve(by=by)
                return True
        return False

    def reject_memo(self, ticker: str, by: str = "Fund Manager") -> bool:
        if not self.pm_output:
            return False
        for memo in self.pm_output.memos:
            if memo.ticker == ticker:
                memo.reject(by=by)
                return True
        return False
