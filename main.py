#!/usr/bin/env python3
"""
main.py — WISDOM-PM CLI Entry Point
=====================================
Usage:
  python main.py run                          # Full 3-step pipeline
  python main.py run --macro                  # Simulate macro shock (anti-panic test)
  python main.py run --data-dir ./trade_data  # Use custom Excel files location
  python main.py score AMBER                  # Score a single stock
  python main.py step1                        # Historical trade analysis only
  python main.py arch                         # Print architecture diagram
  python main.py export                       # Run pipeline and export JSON report
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.prompt import Confirm

from dashboard import (
    console,
    make_progress,
    print_architecture,
    print_header,
    print_memos,
    print_step1,
    print_step2,
    print_step3,
    run_hitl_approval,
)
from orchestrator import WisdomPMOrchestrator

app = typer.Typer(
    name="wisdom-pm",
    help="WISDOM-PM — Intelligent Portfolio Management System",
    add_completion=False,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_pipeline(
    macro_shock: bool = False, 
    skip_hitl: bool = False,
    data_dir: Optional[str] = None,
) -> WisdomPMOrchestrator:
    """Core pipeline runner with Rich progress bar."""
    orch = WisdomPMOrchestrator(macro_shock=macro_shock, data_dir=data_dir)

    with make_progress() as progress:
        from config import PORTFOLIO_STOCKS

        task = progress.add_task(
            "[cyan]Initialising vector store…", total=4 + len(PORTFOLIO_STOCKS) * 2
        )

        # Seed RAG corpus
        progress.update(task, description="[cyan]Seeding analyst corpus (RAG)…")
        _ = orch.vs.collection_size()
        progress.advance(task)

        # Agent 1 — Quant
        for ticker in PORTFOLIO_STOCKS:
            progress.update(task, description=f"[yellow]Agent 1 — Quant: {ticker}…")
            orch.quant_agent.analyse(ticker, macro_shock=macro_shock)
            progress.advance(task)

        # Reload cached outputs into orchestrator
        from agents.quant_analyst import QuantAnalystAgent
        for ticker in PORTFOLIO_STOCKS:
            qn = orch.quant_agent.analyse(ticker, macro_shock=macro_shock)
            orch.quant_outputs[ticker] = qn
            if qn.snapshot:
                orch.snapshots[ticker] = qn.snapshot
            if qn.score_result:
                orch.scores[ticker] = qn.score_result

        # Agent 2 — Qual
        for ticker in PORTFOLIO_STOCKS:
            progress.update(task, description=f"[blue]Agent 2 — Qual: {ticker}…")
            signal = orch.scores.get(ticker, None)
            sig_str = signal.signal if signal else "WATCH"
            ql = orch.qual_agent.research(ticker, quant_signal=sig_str)
            orch.qual_outputs[ticker] = ql
            progress.advance(task)

        # Agent 3 — Risk
        progress.update(task, description="[magenta]Agent 3 — Risk Manager…")
        orch.risk_output = orch.risk_agent.assess(
            scores=orch.scores, snapshots=orch.snapshots, macro_shock=macro_shock
        )
        progress.advance(task)

        # Agent 4 — PM
        progress.update(task, description="[green]Agent 4 — Portfolio Manager: generating memos…")
        orch.pm_output = orch.pm_agent.generate_memos(
            quant_outputs=orch.quant_outputs,
            qual_outputs=orch.qual_outputs,
            risk_output=orch.risk_output,
        )
        progress.advance(task)

        orch.run_timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return orch


# ── CLI Commands ──────────────────────────────────────────────────────────────

@app.command()
def run(
    macro: bool = typer.Option(False, "--macro", "-m", help="Simulate a macro shock (anti-panic test)"),
    no_hitl: bool = typer.Option(False, "--no-hitl", help="Skip interactive approval prompts"),
    export: bool = typer.Option(False, "--export", "-e", help="Export JSON report after run"),
    data_dir: Optional[Path] = typer.Option(
        None, 
        "--data-dir", 
        "-d",
        help="Directory containing Excel trade files (default: ./trade_data)",
    ),
):
    """
    Run the full 3-step WISDOM-PM pipeline and display results.
    """
    print_header()

    if macro:
        console.print("[bold yellow]⚡ MACRO SHOCK MODE — anti-panic locks will engage for eligible stocks.[/]\n")

    data_dir_str = str(data_dir) if data_dir else None
    console.print(f"[dim]Starting WISDOM-PM pipeline… {'(macro shock)' if macro else ''}[/]")
    if data_dir_str:
        console.print(f"[dim]Using trade data from: {data_dir_str}[/]")
    console.print()
    
    t0 = time.time()
    orch = _run_pipeline(macro_shock=macro, data_dir=data_dir_str)
    elapsed = time.time() - t0
    console.print(f"[dim]Pipeline completed in {elapsed:.1f}s[/]\n")

    # ── Display all three steps ───────────────────────────────────────────────
    print_step1(orch)
    print_step2(orch)
    print_step3(orch)
    print_memos(orch.pm_output)
    print_architecture()

    # ── HITL approval ─────────────────────────────────────────────────────────
    if not no_hitl:
        if Confirm.ask("[yellow]Start HITL approval workflow for pending memos?[/]", default=True):
            run_hitl_approval(orch)
            console.print("[green]Approval workflow complete.[/]\n")

    # ── Post-approval status ──────────────────────────────────────────────────
    console.print("\n[bold]Final Memo Status:[/]")
    for memo in orch.pm_output.memos:
        status_str = "PENDING" if memo.approved is None else ("✓ APPROVED" if memo.approved else "✗ REJECTED")
        style = "yellow" if memo.approved is None else ("green" if memo.approved else "red")
        console.print(f"  [{style}]{status_str}[/] {memo.ticker:<10} {memo.recommendation}")

    if export:
        _export_json(orch)


@app.command()
def score(
    ticker: str = typer.Argument(..., help="Stock ticker (AMBER / WELSPUN / ZEEL / DBL)"),
    macro: bool = typer.Option(False, "--macro", "-m"),
):
    """Score a single stock on the WISDOM 5-principle matrix."""
    from config import PORTFOLIO_STOCKS
    from data.market_data import MarketDataFetcher
    from scoring.wisdom_scorer import WisdomScorer

    ticker = ticker.upper()
    if ticker not in PORTFOLIO_STOCKS:
        console.print(f"[red]Unknown ticker: {ticker}. Choose from {list(PORTFOLIO_STOCKS)}[/]")
        raise typer.Exit(1)

    print_header()
    console.print(f"[cyan]Scoring {ticker}…[/]\n")

    fetcher = MarketDataFetcher()
    cfg = PORTFOLIO_STOCKS[ticker]
    snap = fetcher.fetch(ticker, cfg["yf_ticker"], cfg["name"])

    scorer = WisdomScorer()
    result = scorer.score(snap, macro_shock=macro)

    from rich.table import Table
    from rich import box

    table = Table(title=f"WISDOM Score — {ticker}", box=box.ROUNDED)
    table.add_column("Principle", width=28)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Checks", width=55)

    for p in result.principles:
        color = "green" if p.score >= 7 else ("yellow" if p.score >= 5 else "red")
        checks_str = " | ".join(f"{c[0][:28]} {'✓' if c[1] else '✗'}" for c in p.checks[:2])
        table.add_row(p.name, f"[{color}]{p.score:.1f}[/]", checks_str)

    console.print(table)
    from dashboard.terminal_ui import SIGNAL_STYLE
    sig_style = SIGNAL_STYLE.get(result.signal, "white")
    console.print(f"\n  Total WISDOM Score: [bold gold1]{result.total_score:.2f} / 10[/]")
    console.print(f"  Signal: [{sig_style}]{result.signal}[/] — {result.trigger_reason}")
    if result.anti_panic_active:
        console.print("  [yellow]🔒 Anti-panic lock ACTIVE[/]")


@app.command()
def step1(
    data_dir: Optional[Path] = typer.Option(
        None, 
        "--data-dir", 
        "-d",
        help="Directory containing Excel trade files (default: ./trade_data)",
    ),
):
    """Display Step 1: Historical trade analysis and investor bias profile."""
    print_header()
    data_dir_str = str(data_dir) if data_dir else None
    orch = WisdomPMOrchestrator(data_dir=data_dir_str)
    print_step1(orch)


@app.command()
def arch():
    """Display the WISDOM-PM system architecture."""
    print_header()
    print_architecture()


@app.command()
def export_report(
    output: Path = typer.Option(Path("wisdom_pm_report.json"), "--output", "-o"),
    macro: bool = typer.Option(False, "--macro", "-m"),
    data_dir: Optional[Path] = typer.Option(
        None, 
        "--data-dir", 
        "-d",
        help="Directory containing Excel trade files (default: ./trade_data)",
    ),
):
    """Run the pipeline and export a full JSON report."""
    print_header()
    console.print("[cyan]Running pipeline for JSON export…[/]")
    data_dir_str = str(data_dir) if data_dir else None
    orch = _run_pipeline(macro_shock=macro, data_dir=data_dir_str)
    _export_json(orch, output)


def _export_json(orch: WisdomPMOrchestrator, path: Path = Path("wisdom_pm_report.json")) -> None:
    report = orch.to_json()
    path.write_text(report)
    console.print(f"\n[green]✓ Report exported to [bold]{path}[/][/]")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
