"""The six supported training modes and their communication goals."""
MODES = {
    "explain": {"label": "Explain", "goal": "Name the subject, connect its parts and close with a takeaway."},
    "compare": {"label": "Compare", "goal": "Define A and B, compare explicit dimensions and explain consequences."},
    "work_update": {"label": "Work update", "goal": "Lead with status, then explain progress, blockers and next action."},
    "proposal": {"label": "Proposal", "goal": "State the problem, value, tradeoffs and specific ask."},
    "technical_explanation": {"label": "Technical explanation", "goal": "Ground terms and connect components without assuming their relationships."},
    "interview": {"label": "Interview answer", "goal": "Answer every requested part directly and ground the answer."},
}
