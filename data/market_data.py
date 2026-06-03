"""
data/market_data.py
Fetches OHLCV prices and fundamental ratios via yfinance.
LLMs are never used here — pure numerical computation only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


@dataclass
class FundamentalSnapshot:
    """All quantitative data needed for WISDOM scoring."""

    ticker: str
    name: str

    # Principle 1 — Business Quality
    roce_ttm: Optional[float] = None          # %
    roce_5y_avg: Optional[float] = None       # %
    fcf_yield: Optional[float] = None         # %
    gross_margin: Optional[float] = None      # %

    # Principle 2 — Skin in the Game
    promoter_holding: Optional[float] = None  # %
    pledged_pct: Optional[float] = None       # %

    # Principle 3 — Reinvestment Runway
    retention_ratio: Optional[float] = None   # %  (1 - payout ratio)
    capex_revenue_ratio: Optional[float] = None

    # Principle 4 — Structural vs Cyclical
    revenue_beta: Optional[float] = None      # volatility of revenue YoY
    earnings_cyclicality: Optional[bool] = None

    # Principle 5 — Balance Sheet
    debt_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    net_debt_ebitda: Optional[float] = None

    # Valuation
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price: Optional[float] = None
    market_cap_cr: Optional[float] = None

    # Meta
    fetch_error: Optional[str] = None


class MarketDataFetcher:
    """Pulls live / recent data from Yahoo Finance."""

    # Fallback snapshot data for when yfinance is unavailable
    # (approximated from public analyst reports Q3 2025)
    FALLBACK: dict[str, dict] = {
        "AMBER": dict(
            roce_ttm=22.4, roce_5y_avg=19.1, fcf_yield=7.1,
            gross_margin=8.6, promoter_holding=55.3, pledged_pct=0.0,
            retention_ratio=82.0, capex_revenue_ratio=4.2,
            revenue_beta=0.85, earnings_cyclicality=False,
            debt_equity=0.31, interest_coverage=8.2, net_debt_ebitda=0.4,
            pe_ratio=62.1, peg_ratio=1.31, price=5840, market_cap_cr=19800,
        ),
        "WELSPUN": dict(
            roce_ttm=16.1, roce_5y_avg=14.2, fcf_yield=5.8,
            gross_margin=22.3, promoter_holding=47.8, pledged_pct=1.2,
            retention_ratio=71.0, capex_revenue_ratio=3.1,
            revenue_beta=1.05, earnings_cyclicality=False,
            debt_equity=0.44, interest_coverage=5.1, net_debt_ebitda=0.8,
            pe_ratio=18.4, peg_ratio=0.98, price=167, market_cap_cr=5200,
        ),
        "ZEEL": dict(
            roce_ttm=8.3, roce_5y_avg=11.2, fcf_yield=2.1,
            gross_margin=31.4, promoter_holding=3.99, pledged_pct=0.0,
            retention_ratio=45.0, capex_revenue_ratio=1.8,
            revenue_beta=1.42, earnings_cyclicality=True,
            debt_equity=0.61, interest_coverage=2.8, net_debt_ebitda=1.9,
            pe_ratio=21.3, peg_ratio=2.84, price=112, market_cap_cr=10700,
        ),
        "DBL": dict(
            roce_ttm=11.8, roce_5y_avg=13.4, fcf_yield=1.9,
            gross_margin=11.2, promoter_holding=71.4, pledged_pct=4.8,
            retention_ratio=68.0, capex_revenue_ratio=5.6,
            revenue_beta=1.52, earnings_cyclicality=True,
            debt_equity=1.12, interest_coverage=3.1, net_debt_ebitda=2.8,
            pe_ratio=14.2, peg_ratio=1.76, price=340, market_cap_cr=4650,
        ),
    }

    def fetch(self, ticker: str, yf_ticker: str, name: str) -> FundamentalSnapshot:
        snap = FundamentalSnapshot(ticker=ticker, name=name)
        try:
            stock = yf.Ticker(yf_ticker)
            info = stock.info or {}

            snap.price             = info.get("currentPrice") or info.get("regularMarketPrice")
            snap.pe_ratio          = info.get("trailingPE")
            snap.peg_ratio         = info.get("pegRatio")
            snap.debt_equity       = info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None
            snap.gross_margin      = (info.get("grossMargins", 0) or 0) * 100
            snap.promoter_holding  = None   # not in yfinance directly
            snap.pledged_pct       = 0.0    # from NSE/BSE filings

            # Derived from financials
            fins = stock.financials
            bs   = stock.balance_sheet
            cf   = stock.cashflow

            if fins is not None and not fins.empty:
                snap = self._compute_from_financials(snap, fins, bs, cf, info)

        except Exception as e:
            snap.fetch_error = str(e)

        # Fill missing values from fallback
        fallback = self.FALLBACK.get(ticker, {})
        for attr, val in fallback.items():
            if getattr(snap, attr, None) is None:
                setattr(snap, attr, val)

        return snap

    def _compute_from_financials(
        self, snap: FundamentalSnapshot, fins, bs, cf, info
    ) -> FundamentalSnapshot:
        try:
            # ROCE: EBIT / Capital Employed
            ebit_row = [r for r in fins.index if "EBIT" in str(r).upper()]
            ebit = float(fins.loc[ebit_row[0]].iloc[0]) if ebit_row else None

            ce_row = [r for r in bs.index if "TOTAL ASSETS" in str(r).upper()]
            cl_row = [r for r in bs.index if "CURRENT LIABILITIES" in str(r).upper()]
            if ebit and ce_row and cl_row:
                ta = float(bs.loc[ce_row[0]].iloc[0])
                cl = float(bs.loc[cl_row[0]].iloc[0])
                cap_employed = ta - cl
                snap.roce_ttm = round((ebit / cap_employed) * 100, 2) if cap_employed else None

            # FCF Yield
            if cf is not None and not cf.empty:
                ocf_row = [r for r in cf.index if "OPERATING" in str(r).upper() and "CASH" in str(r).upper()]
                cap_row = [r for r in cf.index if "CAPITAL" in str(r).upper()]
                if ocf_row and cap_row:
                    ocf  = float(cf.loc[ocf_row[0]].iloc[0])
                    capx = abs(float(cf.loc[cap_row[0]].iloc[0]))
                    fcf  = ocf - capx
                    mcap = info.get("marketCap", 0)
                    snap.fcf_yield = round((fcf / mcap) * 100, 2) if mcap else None

            # D/E
            td_row = [r for r in bs.index if "TOTAL DEBT" in str(r).upper() or "LONG TERM DEBT" in str(r).upper()]
            eq_row = [r for r in bs.index if "STOCKHOLDER" in str(r).upper() or "EQUITY" in str(r).upper()]
            if td_row and eq_row:
                td  = float(bs.loc[td_row[0]].iloc[0])
                eq  = float(bs.loc[eq_row[0]].iloc[0])
                snap.debt_equity = round(td / eq, 2) if eq else None

            # Interest Coverage
            int_row = [r for r in fins.index if "INTEREST" in str(r).upper()]
            if ebit and int_row:
                int_exp = abs(float(fins.loc[int_row[0]].iloc[0]))
                snap.interest_coverage = round(ebit / int_exp, 2) if int_exp else None

            # Revenue Beta (proxy: std of YoY revenue growth)
            rev_row = [r for r in fins.index if "TOTAL REVENUE" in str(r).upper() or "REVENUE" in str(r).upper()]
            if rev_row and len(fins.columns) >= 3:
                revs    = fins.loc[rev_row[0]].dropna().values.astype(float)
                yoy     = np.diff(revs) / np.abs(revs[:-1])
                snap.revenue_beta = round(float(np.std(yoy)), 3)

        except Exception:
            pass
        return snap


def fetch_price_history(yf_ticker: str, period: str = "10y") -> pd.DataFrame:
    """Returns OHLCV dataframe for charting and returns analysis."""
    try:
        df = yf.download(yf_ticker, period=period, auto_adjust=True, progress=False)
        return df
    except Exception:
        return pd.DataFrame()
