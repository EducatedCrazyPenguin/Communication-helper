"""Explicit synthetic-only calibration against the real managed local provider."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import time

from communication_gym.application.analysis import AnalysisEngine
from communication_gym.application.resources import ResourceCoordinator
from communication_gym.domain.models import GymError
from communication_gym.providers.ollama import OllamaProvider


def calibration_checks(fixture: dict, analysis: dict) -> dict:
    expected = fixture["expected"]
    critic = analysis["critic"]["issues"]
    observed = {issue["kind"] for issue in critic if issue["source"] == "likely_speaker_issue"}
    # An exact quote alone cannot mechanically prove semantic entailment.
    # Record explicit review evidence instead of labelling this a quality pass.
    unknown_roles_preserved = True
    if expected["unknown_battery_roles"]:
        role = re.compile(r"\b(?:main|backup|primary|secondary)\b|मुख्य|बैकअप", re.IGNORECASE)
        unknown_roles_preserved = not any(role.search(item["text"]) for item in analysis["listener"]["clearly_understood"])
    return {"schema_evidence_valid": True, "complete_coverage": analysis["coverage"]["complete"],
            "expected_category_observed": bool(set(expected["communication_categories"]) & observed) if expected["communication_categories"] else None,
            "unknown_battery_roles_preserved": unknown_roles_preserved,
            "uncertainty_not_scored_as_speaker_defect": not any(issue["source"] == "likely_speaker_issue" for issue in critic) if expected["uncertainty_only"] else True,
            "human_entailment_review_required": True, "human_review_fields": ["critic.issues", "listener.clearly_understood", "listener.required_assumptions", "feedback"]}


async def run(args) -> int:
    if not os.environ.get("GYM_DATA_DIR"):
        raise SystemExit("Set GYM_DATA_DIR to a fresh isolated verification root; normal personal storage is not used.")
    root = Path(os.environ["GYM_DATA_DIR"]).resolve()
    source = Path(args.fixtures).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    fixtures = data["fixtures"]
    if len(fixtures) != 18 or {item["language"] for item in fixtures} != {"en", "hi", "hinglish"}:
        raise SystemExit("Calibration requires the complete versioned 18-case fixture set.")
    output = root / "data" / "calibration" / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    provider = OllamaProvider({"managed": True, "model": args.model, "base_url": "http://127.0.0.1:11435", "context_tokens": args.context_tokens})
    resources = ResourceCoordinator(coach=provider)
    engine = AnalysisEngine(provider)
    reports, analyses = [], {}
    try:
        async with resources.lease("coach"):
            probe = await provider.probe()
            if not probe["schema_probe_passed"]:
                raise GymError("schema_probe_failed", "Local model did not pass the explicit schema probe.", status=503)
            for fixture in fixtures:
                started = time.monotonic()
                try:
                    analysis = await engine.analyze(fixture["transcript"], fixture["exercise"], fixture["language"])
                    analyses[fixture["id"]] = analysis
                    checks = calibration_checks(fixture, analysis)
                    record = {"fixture_id": fixture["id"], "status": "succeeded", "seconds": time.monotonic() - started, "checks": checks, "analysis": analysis}
                except GymError as error:
                    record = {"fixture_id": fixture["id"], "status": "failed", "safe_error_code": error.code, "seconds": time.monotonic() - started,
                              "failure": engine.last_failure, "provider_metadata": provider.last_metadata, "completed_window_artifacts": engine.window_artifacts}
                (output / (fixture["id"] + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                reports.append({key: value for key, value in record.items() if key not in {"analysis", "completed_window_artifacts"}})
                print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    finally:
        try:
            await provider.release()
        finally:
            await resources.close()
    ordering = {}
    for language in ("en", "hi", "hinglish"):
        clear = analyses.get(f"{language}-clear_hesitant_comparison")
        ambiguous = analyses.get(f"{language}-ambiguous_comparison")
        checks = {}
        for dimension in ("clarity", "listener_load"):
            a = clear["scores"]["dimensions"].get(dimension) if clear else None
            b = ambiguous["scores"]["dimensions"].get(dimension) if ambiguous else None
            checks[dimension] = a is not None and b is not None and a > b
        ordering[language] = checks
    automatic_pass = len(analyses) == 18 and all(all(values.values()) for values in ordering.values()) and all(
        record.get("checks", {}).get("complete_coverage") and record["checks"]["unknown_battery_roles_preserved"] and record["checks"]["uncertainty_not_scored_as_speaker_defect"] and record["checks"]["expected_category_observed"] is not False for record in reports)
    summary = {"schema_version": 1, "provider": "ollama", "model": args.model, "context_tokens": args.context_tokens, "cases": reports, "clear_hesitant_beats_fluent_ambiguous": ordering,
               "automatic_gate_passed": automatic_pass, "human_review_required": True,
               "release_gate_passed": False, "reason": "Human review of semantic entailment, criticism relevance and natural language is still required; exact quote validation alone is not semantic proof."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"automatic_gate_passed": automatic_pass, "human_review_required": True, "artifact_directory": str(output)}))
    return 0 if automatic_pass else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama"], required=True)
    parser.add_argument("--model", default="qwen3.5:9b-q8_0")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/calibration/v1.json"))
    parser.add_argument("--context-tokens", type=int, default=16384)
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except GymError as error:
        print(json.dumps({"status": "failed", "safe_error_code": error.code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
