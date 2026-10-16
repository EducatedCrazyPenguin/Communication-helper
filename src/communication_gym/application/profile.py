"""Recomputable, evidence-linked aggregates; no personality or diagnostic labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json

DRILLS = {"ambiguous_reference": "Name and define the entity before describing it.", "unnamed_entity": "Name and define the entity before describing it.",
          "implicit_comparison": "Compare A and B using explicitly named dimensions.", "unclear_relationship": "State how the named things relate.",
          "missing_context": "Give a 20-second topic, purpose and scope framing.", "buried_conclusion": "Answer first, then give only supporting detail.",
          "topic_jump": "Connect each new point to the previous one.", "tangent": "Compress the explanation while preserving its main relationship.",
          "repetition": "Compress the explanation while preserving its main relationship.", "weak_proposal": "State the problem, value and specific ask.",
          "weak_call_to_action": "State the problem, value and specific ask."}


def _date(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (AttributeError, ValueError):
        return None


def _eligible(session):
    if session.get("stage") != "complete":
        return False
    first, retry = session.get("initial_analysis"), session.get("retry_analysis")
    if not first or not retry:
        return False
    from communication_gym.application.comparison import comparable
    return comparable(first, retry) and all(analysis.get("coverage", {}).get("complete") is True and analysis.get("status", "succeeded") == "succeeded" for analysis in (first, retry))


def _issues(analysis):
    return {issue["kind"]: issue for issue in analysis.get("critic", {}).get("issues", []) if issue.get("source") == "likely_speaker_issue" and issue.get("confidence") in {"medium", "high"} and (issue.get("evidence") or issue.get("reviewed_range"))}


def compute_profile(sessions: list[dict], *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=90)
    completed = [session for session in sessions if session.get("stage") == "complete" and (date := _date(session.get("completed_at", session.get("updated_at", session.get("created_at"))))) and cutoff <= date <= now]
    completed.sort(key=lambda item: _date(item.get("completed_at", item.get("updated_at", item.get("created_at")))), reverse=True)
    # Deduplicate rehydrated/reanalyzed sessions before the rolling-window cut.
    seen, eligible = set(), []
    for session in completed:
        if session["id"] in seen or not _eligible(session):
            continue
        seen.add(session["id"])
        eligible.append(session)
        if len(eligible) == 20:
            break
    grouped = defaultdict(list)
    for session in eligible:
        provenance = session["initial_analysis"]["provenance"]
        key = (provenance["rubric_version"], provenance["provider"], provenance["model"], session.get("language", "unknown"))
        grouped[key].append(session)
    groups = []
    all_patterns = []
    for key, members in sorted(grouped.items()):
        prevalence, persistence, improved = Counter(), Counter(), Counter()
        after_question_improvement = Counter()
        evidence = defaultdict(list)
        for session in members:
            first, retry = _issues(session["initial_analysis"]), _issues(session["retry_analysis"])
            prevalence.update(first.keys())
            persistence.update(first.keys() & retry.keys())
            # Explicit comparison artifacts, not absence alone, establish improvement.
            session_improved, session_after_question = set(), set()
            answered_targets = set()
            for question in session.get("questions", []):
                resolution = question.get("resolution", question.get("assessment", {}))
                if question.get("status") == "answered" and resolution.get("status") in {"resolved", "partly_resolved"}:
                    answered_targets.update(resolution.get("target_ids", []))
            for change in session.get("comparison", {}).get("issue_changes", []):
                if change.get("status") in {"fixed", "improved"} and change.get("initial_evidence") and change.get("retry_evidence"):
                    matched = next((kind for kind, issue in first.items() if issue["id"] == change.get("issue_id")), None)
                    if matched:
                        session_improved.add(matched)
                        if change["issue_id"] in answered_targets:
                            session_after_question.add(matched)
            improved.update(session_improved)
            after_question_improvement.update(session_after_question)
            for kind, issue in first.items():
                evidence[kind].append({"session_id": session["id"], "issue_id": issue["id"], "analysis_id": session["initial_analysis"]["id"]})
        aggregate_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        patterns = []
        for kind, count in sorted(prevalence.items(), key=lambda pair: (-pair[1], pair[0])):
            confidence = "high" if count >= 5 and len(members) >= 10 else "medium" if count >= 3 and len(members) >= 5 else "insufficient"
            item = {"id": f"{aggregate_id}:{kind}", "kind": kind, "occurrences": count, "eligible_sessions": len(members), "rate": count / len(members),
                    "retry_persistence_count": persistence[kind], "grounded_improvement_count": improved[kind], "after_question_improvement_count": after_question_improvement[kind], "confidence": confidence, "evidence": evidence[kind],
                    "drill": DRILLS.get(kind, "State the main point, then make the missing relationship explicit.")}
            patterns.append(item)
            if confidence != "insufficient":
                all_patterns.append(item)
        strengths = []
        recent = members[:5]
        if len(recent) >= 5:
            for dimension in recent[0]["retry_analysis"]["scores"]["dimensions"]:
                supported = [session for session in recent if (score := session["retry_analysis"]["scores"]["dimensions"].get(dimension)) is not None and score >= 75]
                if len(supported) >= 3:
                    strengths.append({"id": f"{aggregate_id}:strength:{dimension}", "dimension": dimension, "occurrences": len(supported), "eligible_sessions": 5,
                                      "evidence": [{"session_id": s["id"], "analysis_id": s["retry_analysis"]["id"], "criteria": [c for c in s["retry_analysis"]["scores"]["criteria"] if c["dimension"] == dimension]} for s in supported]})
        trends = [{"session_id": s["id"], "date": s.get("completed_at", s.get("updated_at")), "initial": s["initial_analysis"]["scores"]["dimensions"], "retry": s["retry_analysis"]["scores"]["dimensions"]} for s in reversed(members)]
        groups.append({"id": aggregate_id, "rubric_version": key[0], "provider": key[1], "model": key[2], "language": key[3], "eligible_sessions": len(members), "patterns": patterns, "strengths": strengths, "trends": trends})
    focus = max(all_patterns, key=lambda item: (item["rate"], item["occurrences"], item["kind"]), default=None)
    criterion_deltas = defaultdict(list)
    for session in eligible:
        from communication_gym.application.comparison import score_deltas
        for criterion, delta in score_deltas(session["initial_analysis"], session["retry_analysis"])["criteria"].items():
            if delta is not None:
                criterion_deltas[criterion].append(delta)
    best = max(criterion_deltas, key=lambda name: sum(criterion_deltas[name]) / len(criterion_deltas[name]), default=None)
    if best and sum(criterion_deltas[best]) <= 0:
        best = None
    return {"schema_version": 1, "window_days": 90, "maximum_sessions": 20, "eligible_sessions": len(eligible),
            "last_30_day_session_count": len({s["id"] for s in completed if _date(s.get("completed_at", s.get("updated_at", s.get("created_at")))) >= now - timedelta(days=30)}),
            "groups": groups, "active_focus": focus, "most_improved_criterion": {"id": best, "mean_level_delta": sum(criterion_deltas[best]) / len(criterion_deltas[best]), "sample_count": len(criterion_deltas[best])} if best else None,
            "insufficient_data": not all_patterns, "message": "Not enough comparable sessions for recurring patterns." if not all_patterns else "Patterns are observed rates, not diagnoses."}
