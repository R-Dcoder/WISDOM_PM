"""
dashboard/terminal_ui.py
Rich-powered terminal dashboard for WISDOM-PM.
Renders all 3 steps, scores, memos, and HITL approval workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from orchestrator import WisdomPMOrchestrator
    from agents.portfolio_manager import PMOrchestrationOutput, TradeRecommendationMemo
    from agents.risk_manager import RiskAssessmentOutput

console = Console()

# ── Colour palette ─────────────────────────────────────────────────────────────
SIGNAL_STYLE = {
    "BUY":   "bold green",
    "HOLD":  "bold blue",
    "SELL":  "bold red",
    "WATCH": "bold yellow",
}
SEV_STYLE = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "dim cyan"}


# ── Header ─────────────────────────────────────────────────────────────────────

def print_header() -> None:
    console.print()
    console.print(
        Panel(
            Align.center(
                Text.assemble(
                    ("  WISDOM–PM  ", "bold gold1 on black"),
                    ("\n", ""),
                    ("Intelligent Portfolio Management System", "dim"),
                    ("\nQuant · RAG · Multi-Agent · Human-in-the-Loop", "italic dim"),
                )
            ),
            border_style="gold1",
            padding=(1, 4),
        )
    )


# ── Step 1: Investor Profile ────────────────────────────────────────────────────

def print_step1(orch: "WisdomPMOrchestrator") -> None:
    console.print(Rule("[bold gold1]STEP 1 — Historical Trade Analysis & Investor Profiling[/]"))

    from config import HISTORICAL_TRADES, IDENTIFIED_BIASES

    # Trade history table
    table = Table(title="10-Year Trade History", box=box.SIMPLE_HEAVY, border_style="dim")
    table.add_column("Ticker",  style="bold cyan",    width=10)
    table.add_column("Action",  justify="center",     width=8)
    table.add_column("Year",    justify="center",     width=6)
    table.add_column("Price ₹", justify="right",      width=10)
    table.add_column("Qty",     justify="right",      width=7)
    table.add_column("Thesis / Note",                 width=55)

    action_style = {"BUY": "green", "SELL": "red", "HOLD": "blue"}

    for t in HISTORICAL_TRADES:
        bias_tag = f"  [red][BIAS: {t['bias'].upper()}][/]" if "bias" in t else ""
        table.add_row(
            t["ticker"],
            f"[{action_style.get(t['action'], 'white')}]{t['action']}[/]",
            str(t["year"]),
            f"{t['price']:,}",
            str(t["qty"]),
            t["thesis"][:55] + bias_tag,
        )
    console.print(table)
    console.print()

    # Bias profile
    bias_table = Table(title="Identified Investor Biases", box=box.ROUNDED, border_style="red")
    bias_table.add_column("Bias",         style="bold yellow", width=28)
    bias_table.add_column("Stocks",       width=14)
    bias_table.add_column("Mitigation",   width=52)

    for bias_id, b in IDENTIFIED_BIASES.items():
        bias_table.add_row(
            b["name"],
            ", ".join(b["affected_stocks"]),
            b["mitigation"][:52],
        )
    console.print(bias_table)
    console.print()


# ── Step 2: WISDOM Scores ────────────────────────────────────────────────────────

def print_step2(orch: "WisdomPMOrchestrator") -> None:
    console.print(Rule("[bold gold1]STEP 2 — WISDOM Decision Matrix[/]"))

    for ticker, qn in orch.quant_outputs.items():
        if not qn.score_result:
            continue
        r   = qn.score_result
        ql  = orch.qual_outputs.get(ticker)
        s   = qn.snapshot

        sig_style = SIGNAL_STYLE.get(r.signal, "white")

        # Header panel
        header_text = Text.assemble(
            (f"  {ticker}  ", "bold white on dark_blue"),
            ("  ", ""),
            (f"{r.total_score:.1f} / 10", "bold gold1"),
            ("  →  ", "dim"),
            (f"{r.signal}", sig_style),
        )
        if r.anti_panic_active:
            header_text.append("  🔒 ANTI-PANIC LOCK", style="bold yellow")
        console.print(Panel(header_text, expand=False))

        # Principles table
        p_table = Table(box=box.MINIMAL, show_header=True, border_style="dim")
        p_table.add_column("Principle", width=26)
        p_table.add_column("Score", justify="center", width=10)
        p_table.add_column("Bar",   width=22)
        p_table.add_column("Top Check", width=45)

        for p in r.principles:
            bar_filled = int(p.score / 10 * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            color = "green" if p.score >= 7 else ("yellow" if p.score >= 5 else "red")
            top_check = p.checks[0][2] if p.checks else ""
            p_table.add_row(
                p.name,
                f"[{color}]{p.score:.1f}[/]",
                f"[{color}]{bar}[/]",
                top_check,
            )
        console.print(p_table)

        # Quant snapshot
        if s:
            snap_items = [
                f"ROCE: {s.roce_ttm:.1f}%" if s.roce_ttm else "",
                f"FCF Yield: {s.fcf_yield:.1f}%" if s.fcf_yield else "",
                f"D/E: {s.debt_equity:.2f}" if s.debt_equity else "",
                f"IC: {s.interest_coverage:.1f}x" if s.interest_coverage else "",
                f"Promoter: {s.promoter_holding:.1f}%" if s.promoter_holding else "",
                f"PEG: {s.peg_ratio:.2f}" if s.peg_ratio else "",
            ]
            console.print("  Quant: " + "  |  ".join(x for x in snap_items if x))

        # Qual summary
        if ql:
            sent_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(ql.analyst_sentiment, "white")
            console.print(
                f"  Qual: Sentiment=[{sent_color}]{ql.analyst_sentiment.upper()}[/]  "
                f"Thesis={'[green]INTACT[/]' if ql.thesis_intact else '[red]BROKEN[/]'}  "
                f"Docs retrieved: {len(ql.retrieved_chunks)}"
            )

        console.print()


# ── Step 3: Risk + Memos ─────────────────────────────────────────────────────────

def print_step3(orch: "WisdomPMOrchestrator") -> None:
    console.print(Rule("[bold gold1]STEP 3 — Risk Assessment & Trade Recommendation Memos[/]"))

    risk = orch.risk_output
    if risk:
        r_color = "green" if risk.portfolio_risk_score < 4 else ("yellow" if risk.portfolio_risk_score < 7 else "red")
        console.print(
            f"  Portfolio Risk Score: [{r_color}]{risk.portfolio_risk_score}/10[/]  "
            f"Cyclical Exposure: {risk.cyclical_exposure_pct:.0f}%  "
            f"Max Concentration: {risk.max_concentration_pct:.0f}%"
        )
        if risk.flags:
            risk_table = Table(title="Risk Flags", box=box.SIMPLE, border_style="dim")
            risk_table.add_column("Severity", width=10)
            risk_table.add_column("Category", width=14)
            risk_table.add_column("Stock",    width=10)
            risk_table.add_column("Description", width=55)
            for f in sorted(risk.flags, key=lambda x: ["HIGH","MEDIUM","LOW"].index(x.severity)):
                risk_table.add_row(
                    f"[{SEV_STYLE.get(f.severity, 'white')}]{f.severity}[/]",
                    f.category, f.stock or "—", f.description[:55],
                )
            console.print(risk_table)
        console.print()


def print_memos(pm_output: "PMOrchestrationOutput") -> None:
    console.print(Rule("[bold gold1]HUMAN-IN-THE-LOOP — Trade Recommendation Memos[/]"))
    console.print(
        Panel(
            "[yellow]⚠  No trade is executed automatically.[/]  "
            "Each memo below requires explicit [bold]fund manager approval[/] via the CLI prompt.",
            border_style="yellow",
        )
    )
    console.print()

    for memo in sorted(pm_output.memos, key=lambda m: ["HIGH","MEDIUM","LOW"].index(m.urgency)):
        sig_style = SIGNAL_STYLE.get(memo.recommendation, "white")
        urg_style = SEV_STYLE.get(memo.urgency, "white")

        tree = Tree(
            Text.assemble(
                (f"[{memo.urgency}] ", urg_style),
                (memo.ticker, "bold cyan"),
                " — ",
                (memo.recommendation, sig_style),
                f"  WISDOM {memo.wisdom_score:.1f}/10",
            )
        )
        synth_branch = tree.add("[dim]Agent Syntheses[/]")
        for s in memo.agent_syntheses:
            synth_branch.add(Text(s[:95], style="dim"))

        tree.add(Text(f"Rationale : {memo.rationale[:100]}", style="italic"))
        tree.add(Text(f"Instruction: {memo.instruction[:100]}", style="bold white"))
        tree.add(Text(f"Risk Note  : {memo.risk_note[:100]}", style="yellow"))

        if memo.bias_override_applied:
            tree.add("[red]⚡ Bias override applied — WISDOM discipline enforced[/]")
        if memo.anti_panic_active:
            tree.add("[yellow]🔒 Anti-panic lock active — portfolio frozen[/]")

        status = (
            "[dim]PENDING APPROVAL[/]" if memo.approved is None
            else ("[green]✓ APPROVED[/]" if memo.approved else "[red]✗ REJECTED[/]")
        )
        tree.add(f"Status: {status}  |  Generated: {memo.generated_at}")

        console.print(Panel(tree, border_style="dim"))
    console.print()


# ── HITL Approval Workflow ────────────────────────────────────────────────────────

def run_hitl_approval(orch: "WisdomPMOrchestrator") -> None:
    console.print(Rule("[bold gold1]HUMAN-IN-THE-LOOP APPROVAL WORKFLOW[/]"))
    pending = orch.pm_output.pending_memos if orch.pm_output else []

    if not pending:
        console.print("[green]All memos already reviewed.[/]")
        return

    console.print(f"[yellow]{len(pending)} memo(s) pending approval.[/]")
    console.print()

    for memo in pending:
        sig_style = SIGNAL_STYLE.get(memo.recommendation, "white")
        console.print(
            Panel(
                Text.assemble(
                    (f"{memo.ticker}  ", "bold cyan"),
                    (memo.recommendation, sig_style),
                    (f"  WISDOM {memo.wisdom_score:.1f}  ", "gold1"),
                    (f"[{memo.urgency}]", SEV_STYLE.get(memo.urgency, "white")),
                    "\n\n",
                    (f"Instruction: {memo.instruction[:120]}", "bold white"),
                    "\n",
                    (f"Risk: {memo.risk_note[:100]}", "yellow"),
                ),
                title=f"Memo: {memo.ticker}",
                border_style="blue",
            )
        )
        choice = Prompt.ask(
            f"  Approve this {memo.recommendation} recommendation for [cyan]{memo.ticker}[/]?",
            choices=["y", "n", "s"],
            default="s",
        )
        name = "Fund Manager"
        if choice == "y":
            memo.approve(by=name)
            console.print(f"  [green]✓ Memo approved by {name}. Order queued.[/]\n")
        elif choice == "n":
            memo.reject(by=name)
            console.print(f"  [red]✗ Memo rejected. Returned to analysis queue.[/]\n")
        else:
            console.print(f"  [dim]Skipped. Memo remains pending.[/]\n")


# ── Architecture summary ──────────────────────────────────────────────────────────

def print_architecture() -> None:
    console.print(Rule("[bold gold1]ARCHITECTURE — WISDOM-PM Stack[/]"))

    layers = [
        ("Data Ingestion",     "Yahoo Finance / NSE APIs · Screener.in · TickerTape",     "cyan"),
        ("Unstructured Data",  "Analyst PDFs (Kotak, JM, Anand Rathi) · Concall transcripts", "cyan"),
        ("ETL / Processing",   "AWS Lambda DAGs · Daily OHLCV · Quarterly financials",    "blue"),
        ("Vector DB",          "ChromaDB (Pinecone in prod) · Chunked embeddings for RAG","blue"),
        ("Secure Storage",     "Encrypted PostgreSQL · Trade history NEVER sent to LLMs", "green"),
        ("Agent 1 — Quant",    "Tool-Calling: LLM calls Python/Pandas — never computes numbers itself", "yellow"),
        ("Agent 2 — Qual",     "RAG retrieval · Sentiment · Thesis-break detection",      "yellow"),
        ("Agent 3 — Risk",     "Portfolio concentration · Cyclical exposure · Governance","yellow"),
        ("Agent 4 — PM",       "Synthesis → Trade Memo → Human sign-off (HITL dashboard)","gold1"),
        ("Hallucination Guard","LLM prohibited from arithmetic — Function Calling only",  "red"),
        ("Cloud",              "AWS S3 · RDS · Pinecone · Lambda · EventBridge",          "dim"),
    ]

    table = Table(box=box.SIMPLE_HEAVY, border_style="dim", show_header=False)
    table.add_column("Layer",       style="bold", width=26)
    table.add_column("Description", width=68)

    for layer, desc, color in layers:
        table.add_row(f"[{color}]{layer}[/]", desc)

    console.print(table)
    console.print()


# ── Progress spinner ──────────────────────────────────────────────────────────────

def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        console=console,
        transient=True,
    )
