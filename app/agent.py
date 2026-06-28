from __future__ import annotations
import json
import base64
from typing import Any
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.workflow import Edge, Workflow, node, START
from google.adk.events.event import Event
from google.adk.agents.context import Context
from app.config import EXPENSE_THRESHOLD, GEMINI_MODEL


@node
def parse_expense(ctx: Context, node_input: Any):
    """Parse incoming expense JSON."""
    try:
        if isinstance(node_input, dict) and "data" in node_input:
            raw = node_input["data"]
            try:
                payload = json.loads(base64.b64decode(raw))
            except Exception:
                payload = raw if isinstance(raw, dict) else json.loads(raw)
        else:
            payload = node_input if isinstance(node_input, dict) else json.loads(str(node_input))

        expense = {
            "amount": float(payload.get("amount", 0)),
            "submitter": payload.get("submitter", "unknown"),
            "category": payload.get("category", "general"),
            "description": payload.get("description", ""),
            "date": payload.get("date", ""),
        }
        yield Event(data=expense, state={"expense": expense})
    except Exception as e:
        yield Event(data={"error": str(e)})


@node
def route_by_amount(ctx: Context, node_input: Any):
    """Route: under $100 -> auto, $100+ -> review."""
    expense = ctx.state.get("expense", {})
    amount = expense.get("amount", 0)
    if amount < EXPENSE_THRESHOLD:
        yield Event(data=expense, route="auto")
    else:
        yield Event(data=expense, route="review")


@node
def auto_approve(ctx: Context, node_input: Any):
    """Auto-approve low-value expenses."""
    expense = ctx.state.get("expense", {})
    result = {
        "status": "AUTO_APPROVED",
        "expense": expense,
        "reason": "Amount is below $100 threshold.",
    }
    yield Event(data=result, state={"outcome": result})


@node
def llm_review(ctx: Context, node_input: Any):
    """Use Gemini to review high-value expenses."""
    expense = ctx.state.get("expense", {})
    agent = Agent(
        name="reviewer",
        model=Gemini(model=GEMINI_MODEL),
        instruction="""You are a corporate expense compliance officer.
Analyze the expense and identify risk factors.
Respond with a brief risk summary and recommendation: APPROVE, FLAG, or REJECT.""",
    )
    prompt = f"Review this expense: {json.dumps(expense)}"
    response = agent.generate_content(prompt)
    summary = response.text if hasattr(response, "text") else str(response)
    yield Event(
        data={"expense": expense, "llm_summary": summary},
        state={"llm_summary": summary},
    )


@node
def human_review(ctx: Context, node_input: Any):
    """Record outcome after human reviews the LLM summary."""
    expense = ctx.state.get("expense", {})
    llm_summary = ctx.state.get("llm_summary", "No summary available.")
    result = {
        "status": "PENDING_HUMAN_REVIEW",
        "expense": expense,
        "llm_summary": llm_summary,
        "message": "Please review the above and approve or reject manually.",
    }
    yield Event(data=result, state={"outcome": result})


root_agent = Workflow(
    name="expense_workflow",
    edges=[
        Edge(from_node=START, to_node=parse_expense),
        Edge(from_node=parse_expense, to_node=route_by_amount),
        Edge(from_node=route_by_amount, to_node=auto_approve, route="auto"),
        Edge(from_node=route_by_amount, to_node=llm_review, route="review"),
        Edge(from_node=llm_review, to_node=human_review),
    ],
)

app = App(
    name="app",
    root_agent=root_agent,
)