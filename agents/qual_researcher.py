"""
agents/qual_researcher.py
Agent 2 — Qualitative Researcher
Queries the Vector DB for analyst reports and concall sentiment.
Uses the LLM ONLY for interpreting text — never for computing numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, IDENTIFIED_BIASES, PORTFOLIO_STOCKS
from rag.vector_store import RetrievedChunk, WisdomVectorStore

QUAL_TOOLS: List[Dict] = [
    {
        "name": "search_analyst_reports",
        "description": (
            "Semantic search over analyst PDFs and broker notes for a given stock. "
            "Returns the most relevant text chunks with source attribution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "query":  {"type": "string", "description": "What to look for, e.g. 'moat pricing power margins'"},
                "n_results": {"type": "integer", "default": 4},
            },
            "required": ["ticker", "query"],
        },
    },
    {
        "name": "search_concall_transcripts",
        "description": (
            "Search earnings concall transcripts for management tone changes, "
            "capital allocation commentary, or guidance shifts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "query":  {"type": "string"},
            },
            "required": ["ticker", "query"],
        },
    },
    {
        "name": "detect_thesis_break",
        "description": (
            "Check whether the original investment thesis for a stock has been broken. "
            "Returns a structured flag with evidence from retrieved documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_investor_bias_profile",
        "description": "Return known behavioural biases relevant to this stock ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
]


@dataclass
class QualResearchOutput:
    ticker: str
    analyst_sentiment: str = "neutral"    # bullish | bearish | neutral
    thesis_intact: bool = True
    thesis_break_evidence: List[str] = field(default_factory=list)
    key_positives: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    bias_flags: List[str] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    agent_summary: str = ""
    tool_calls_made: List[str] = field(default_factory=list)


class QualResearcherAgent:
    """
    Agent 2: Qualitative Researcher
    Interprets analyst reports and concall transcripts via RAG + LLM.
    The LLM reads text and identifies sentiment — it never computes ratios.
    """

    def __init__(self, vector_store: Optional[WisdomVectorStore] = None):
        self.vs = vector_store or WisdomVectorStore()
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    # ── Public entry point ────────────────────────────────────────────────────

    def research(self, ticker: str, quant_signal: str = "WATCH") -> QualResearchOutput:
        output = QualResearchOutput(ticker=ticker)

        if not self.client:
            return self._offline_research(ticker, output)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Perform qualitative research for {ticker}. "
                    f"The quant model has flagged this stock as: {quant_signal}. "
                    "Use the tools to: "
                    "1) Search analyst reports for moat, pricing power, and risks. "
                    "2) Search concall transcripts for management tone and guidance. "
                    "3) Detect if the investment thesis has broken. "
                    "4) Get investor bias profile for this stock. "
                    "Then write a structured qualitative summary with: "
                    "SENTIMENT (bullish/bearish/neutral), "
                    "THESIS STATUS (intact/broken), "
                    "KEY POSITIVES (2-3 bullets), "
                    "KEY RISKS (2-3 bullets). "
                    "Base all conclusions only on the retrieved text — do not invent facts."
                ),
            }
        ]

        max_loops = 6
        loop_count = 0
        while loop_count < max_loops:
            loop_count += 1
            resp = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=2000,
                tools=QUAL_TOOLS,
                messages=messages,
            )

            text_parts = [b.text for b in resp.content if b.type == "text"]
            if text_parts:
                raw = "\n".join(text_parts)
                output.agent_summary = raw
                output = self._parse_summary(output, raw)

            if resp.stop_reason == "end_turn":
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                output.tool_calls_made.append(block.name)
                result = self._dispatch_tool(block.name, block.input, ticker, output)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user",      "content": tool_results})

        return output

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, inputs: dict, ticker: str, output: QualResearchOutput) -> dict:
        t = inputs.get("ticker", ticker)
        if name == "search_analyst_reports":
            chunks = self.vs.query(inputs.get("query", ""), ticker=t, n_results=inputs.get("n_results", 4))
            output.retrieved_chunks.extend(chunks)
            return {"chunks": [{"source": c.source, "date": c.date, "text": c.text[:600]} for c in chunks]}

        if name == "search_concall_transcripts":
            chunks = self.vs.query(inputs.get("query", "") + " management concall", ticker=t, n_results=3)
            output.retrieved_chunks.extend(chunks)
            return {"chunks": [{"source": c.source, "date": c.date, "text": c.text[:600]} for c in chunks]}

        if name == "detect_thesis_break":
            return self._tool_detect_thesis_break(t)

        if name == "get_investor_bias_profile":
            return self._tool_get_bias_profile(t)

        return {"error": f"Unknown tool {name}"}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_detect_thesis_break(self, ticker: str) -> dict:
        """Rule-based thesis break detection from corpus."""
        chunks = self.vs.query(f"{ticker} structural decline deterioration bearish sell", ticker=ticker, n_results=4)
        bearish_keywords = {"structural decline", "thesis break", "sell", "reduce", "deterioration",
                            "no catalyst", "impaired", "cyclical", "sell trigger"}
        break_evidence = []
        for c in chunks:
            low = c.text.lower()
            hits = [kw for kw in bearish_keywords if kw in low]
            if hits:
                break_evidence.append(f"[{c.source}] {c.text[:200]}...")
        return {
            "thesis_broken": len(break_evidence) >= 2,
            "evidence_count": len(break_evidence),
            "evidence_snippets": break_evidence[:3],
        }

    def _tool_get_bias_profile(self, ticker: str) -> dict:
        relevant = {
            bias_id: bias_info
            for bias_id, bias_info in IDENTIFIED_BIASES.items()
            if ticker in bias_info.get("affected_stocks", [])
        }
        return {
            "ticker": ticker,
            "biases": [
                {"id": k, "name": v["name"], "mitigation": v["mitigation"]}
                for k, v in relevant.items()
            ],
        }

    # ── Offline mode ──────────────────────────────────────────────────────────

    def _offline_research(self, ticker: str, output: QualResearchOutput) -> QualResearchOutput:
        """Fallback when no API key — uses retrieved chunks + rule-based analysis."""
        chunks = self.vs.query(f"{ticker} fundamentals outlook", ticker=ticker, n_results=4)
        output.retrieved_chunks = chunks

        bearish_kw = {"decline", "sell", "reduce", "deterioration", "no catalyst", "cyclical", "risk"}
        bullish_kw = {"buy", "target", "growth", "structural", "tailwind", "improve", "strong", "expand"}

        b_score = sum(1 for c in chunks for w in bearish_kw if w in c.text.lower())
        u_score = sum(1 for c in chunks for w in bullish_kw if w in c.text.lower())

        output.analyst_sentiment = "bullish" if u_score > b_score else ("bearish" if b_score > u_score else "neutral")
        output.thesis_intact = b_score < 4

        break_result = self._tool_detect_thesis_break(ticker)
        output.thesis_intact = not break_result.get("thesis_broken", False)
        output.thesis_break_evidence = break_result.get("evidence_snippets", [])

        bias = self._tool_get_bias_profile(ticker)
        output.bias_flags = [b["id"] for b in bias.get("biases", [])]

        output.agent_summary = (
            f"[Offline] {ticker}: sentiment={output.analyst_sentiment}, "
            f"thesis_intact={output.thesis_intact}, "
            f"docs_retrieved={len(chunks)}"
        )

        if chunks:
            output.key_positives = [c.text[:120] + "..." for c in chunks[:2] if u_score > 0]
            output.key_risks     = [c.text[:120] + "..." for c in chunks[:2] if b_score > 0]

        return output

    def _parse_summary(self, output: QualResearchOutput, text: str) -> QualResearchOutput:
        low = text.lower()
        if "sentiment: bullish" in low or "bullish" in low:
            output.analyst_sentiment = "bullish"
        elif "sentiment: bearish" in low or "bearish" in low:
            output.analyst_sentiment = "bearish"
        else:
            output.analyst_sentiment = "neutral"
        output.thesis_intact = "thesis status: broken" not in low and "thesis broken" not in low
        return output
