"""Cross-exam target accounting; persistence and stage changes belong to sessions."""
from communication_gym.domain.models import GymError


def open_question(history: list[dict]) -> dict | None:
    opened = [question for question in history if question.get("status") == "open"]
    if len(opened) > 1:
        raise GymError("invalid_question_state", "More than one persisted question is open.", status=409)
    return opened[0] if opened else None


def unresolved_targets(analysis: dict, history: list[dict]) -> list[str]:
    resolved = set()
    for question in history:
        assessment = question.get("resolution", question.get("assessment", {}))
        if question.get("status") == "answered" and assessment.get("status") == "resolved":
            resolved.update(assessment.get("target_ids", []))
    targets = [issue["id"] for issue in analysis.get("feedback", []) if issue.get("source") != "possible_transcription_error"]
    for field in ("ambiguities", "missing_relationships", "ungrounded_terms", "confusion_events"):
        targets.extend(item["id"] for item in analysis.get("listener", {}).get(field, []) if item.get("id"))
    return list(dict.fromkeys(target for target in targets if target not in resolved))


def validate_resolution(result: dict, question: dict):
    if set(result["target_ids"]) - set(question["target_ids"]):
        raise ValueError("resolution cites unknown targets")
    if result["status"] == "resolved" and (not result["answer_evidence"] or set(result["target_ids"]) != set(question["target_ids"])):
        raise ValueError("resolution requires affirmative answer evidence for every target")
    if result["status"] == "resolved":
        spoken = " ".join(ref.get("quote", "") for ref in result["answer_evidence"])
        tokens = {token.casefold().strip(".,!?।;:…") for token in spoken.split()} - {""}
        acknowledgements = {"yes", "yeah", "yep", "correct", "haan", "han", "ji", "हाँ", "हां", "जी", "सही", "ठीक"}
        if not tokens or tokens <= acknowledgements:
            raise ValueError("acknowledgement alone supplies no missing relationship")
    return result
