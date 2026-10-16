from copy import deepcopy
from datetime import datetime, timedelta, timezone

from communication_gym.application.profile import compute_profile

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def session(index, has_issue=True, model="model", language="en"):
    issue = {"id": f"issue-{index}", "kind": "ambiguous_reference", "source": "likely_speaker_issue", "confidence": "high", "evidence": [{"segment_id": "s"}]}
    analysis = {"id": f"analysis-{index}", "status": "succeeded", "coverage": {"complete": True}, "provenance": {"provider": "local", "model": model, "rubric_version": "1", "prompt_version": "1", "schema_version": "1", "context_tokens": 16384},
                "critic": {"issues": [issue] if has_issue else []}, "scores": {"dimensions": {"clarity": 75}, "criteria": [{"id": "relations_explicit", "dimension": "clarity", "level": 1, "evidence": [{"segment_id": "s"}]}]}}
    return {"id": str(index), "stage": "complete", "created_at": (NOW - timedelta(days=index)).isoformat(), "language": language, "initial_analysis": analysis, "retry_analysis": deepcopy(analysis), "comparison": {"issue_changes": []}}


def test_three_of_three_insufficient_but_three_of_five_emerging():
    assert compute_profile([session(i) for i in range(3)], now=NOW)["active_focus"] is None
    result = compute_profile([session(i, i < 3) for i in range(5)], now=NOW)
    assert result["active_focus"]["confidence"] == "medium"
    assert result["active_focus"]["occurrences"] == 3
    assert result["active_focus"]["eligible_sessions"] == 5
    assert result["active_focus"]["rate"] == .6
    assert len(result["active_focus"]["evidence"]) == 3


def test_five_of_ten_high_confidence_without_duplicate_reanalysis():
    sessions = [session(i, i < 5) for i in range(10)]
    result = compute_profile(sessions + [deepcopy(sessions[0])], now=NOW)
    assert result["eligible_sessions"] == 10
    assert result["active_focus"]["confidence"] == "high"
    assert result["active_focus"]["occurrences"] == 5


def test_correction_deletion_and_incomplete_analyses_remove_contributions():
    sessions = [session(i, i < 3) for i in range(5)]
    assert compute_profile(sessions, now=NOW)["active_focus"]
    sessions[0]["initial_analysis"]["critic"]["issues"] = []
    assert compute_profile(sessions, now=NOW)["active_focus"] is None
    sessions[1]["retry_analysis"]["coverage"]["complete"] = False
    assert compute_profile(sessions, now=NOW)["eligible_sessions"] == 4
    assert compute_profile(sessions[2:], now=NOW)["eligible_sessions"] == 3


def test_possible_asr_and_cross_exam_do_not_create_weaknesses():
    sessions = [session(i) for i in range(5)]
    for item in sessions:
        item["initial_analysis"]["critic"]["issues"][0]["source"] = "possible_transcription_error"
        item["cross_exam_analysis"] = session(30)["initial_analysis"]
    assert compute_profile(sessions, now=NOW)["active_focus"] is None


def test_provider_language_groups_and_rolling_cutoff_are_visible():
    sessions = [session(i, model="first" if i < 3 else "second", language="hi" if i < 3 else "hinglish") for i in range(6)]
    sessions.append(session(100))
    profile = compute_profile(sessions, now=NOW)
    assert profile["eligible_sessions"] == 6
    assert {(group["model"], group["language"]) for group in profile["groups"]} == {("first", "hi"), ("second", "hinglish")}
    assert profile["active_focus"] is None


def test_maximum_twenty_sessions_and_strength_threshold():
    profile = compute_profile([session(i) for i in range(25)], now=NOW)
    assert profile["eligible_sessions"] == 20
    assert profile["groups"][0]["strengths"][0]["occurrences"] == 5

def test_after_question_improvement_is_grounded_and_counted_once_per_session():
    sessions = [session(i) for i in range(5)]
    change = {"issue_id": "issue-0", "status": "improved", "initial_evidence": [{"quote": "that"}], "retry_evidence": [{"quote": "Battery A"}]}
    sessions[0]["comparison"]["issue_changes"] = [change, deepcopy(change)]
    sessions[0]["questions"] = [{"status": "answered", "resolution": {"status": "partly_resolved", "target_ids": ["issue-0"]}}]
    focus = compute_profile(sessions, now=NOW)["active_focus"]
    assert focus["grounded_improvement_count"] == 1
    assert focus["after_question_improvement_count"] == 1
