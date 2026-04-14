from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from profiles import (
    build_chat_model,
    build_embedding_model,
    get_default_profile_name,
    get_profile_names,
    resolve_profile,
)
from runtime import PROJECT_ROOT


DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases" / "smoke_cases.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "runs" / "latest_results.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Cases JSON must be a list.")
    return cases


def _contains_substring(values: list[str], expected: list[str]) -> tuple[bool, list[str]]:
    if not expected:
        return True, []
    matched = []
    lowered_values = [value.lower() for value in values]
    for candidate in expected:
        lowered_candidate = candidate.lower()
        if any(lowered_candidate in value for value in lowered_values):
            matched.append(candidate)
    return bool(matched), matched


def _contains_all(answer: str, expected_parts: list[str]) -> tuple[bool, list[str]]:
    if not expected_parts:
        return True, []
    missing = [part for part in expected_parts if part not in answer]
    return not missing, missing


def _contains_none(answer: str, forbidden_parts: list[str]) -> tuple[bool, list[str]]:
    if not forbidden_parts:
        return True, []
    found = [part for part in forbidden_parts if part in answer]
    return not found, found


def evaluate_case(case: dict[str, Any], profile_name: str) -> dict[str, Any]:
    from back import answer_question, clear_session_history, summarize_contexts

    case_id = case.get("id") or case["question"]
    session_id = f"harness::{case_id}"
    clear_session_history(session_id)
    turn = answer_question(
        case["question"],
        session_id=session_id,
        commit_history=False,
        profile_name=profile_name,
    )
    retrieval_summary = summarize_contexts(turn.retrieved_context)

    title_hit, matched_titles = _contains_substring(
        retrieval_summary["titles"],
        case.get("expected_titles", []),
    )
    section_hit, matched_sections = _contains_substring(
        retrieval_summary["sections"],
        case.get("expected_sections", []),
    )
    must_contain_ok, missing_parts = _contains_all(
        turn.answer,
        case.get("must_contain", []),
    )
    must_not_contain_ok, found_forbidden = _contains_none(
        turn.answer,
        case.get("must_not_contain", []),
    )

    min_contexts = int(case.get("min_contexts", 1))
    max_contexts = case.get("max_contexts")
    min_contexts_ok = retrieval_summary["context_count"] >= min_contexts
    max_contexts_ok = True if max_contexts is None else retrieval_summary["context_count"] <= int(max_contexts)
    no_table_ok = (
        retrieval_summary["table_count"] == 0
        if case.get("forbid_is_table")
        else True
    )
    no_quote_ok = (
        retrieval_summary["quote_count"] == 0
        if case.get("forbid_is_quote")
        else True
    )

    checks = {
        "title_hit": title_hit,
        "section_hit": section_hit,
        "min_contexts_ok": min_contexts_ok,
        "max_contexts_ok": max_contexts_ok,
        "no_table_ok": no_table_ok,
        "no_quote_ok": no_quote_ok,
        "must_contain_ok": must_contain_ok,
        "must_not_contain_ok": must_not_contain_ok,
    }
    passed = all(checks.values())

    return {
        "case_id": case_id,
        "category": case.get("category", "general"),
        "question": turn.question,
        "normalized_question": turn.normalized_question,
        "answer": turn.answer,
        "latency_ms": turn.latency_ms,
        "profile_name": turn.profile_name,
        "chat_model": turn.chat_model,
        "embedding_model": turn.embedding_model,
        "search_filter": turn.search_filter,
        "retrieval": {
            **retrieval_summary,
            "matched_titles": matched_titles,
            "matched_sections": matched_sections,
        },
        "checks": checks,
        "missing_parts": missing_parts,
        "forbidden_parts_found": found_forbidden,
        "passed": passed,
        "ground_truth": case.get("ground_truth", ""),
        "contexts": [item["text"] for item in turn.retrieved_context],
        "retrieved_context": turn.retrieved_context,
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0.0,
        }

    return {
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "pass_rate": round(
            sum(1 for result in results if result["passed"]) / len(results),
            4,
        ),
        "avg_latency_ms": round(mean(result["latency_ms"] for result in results), 2),
        "title_hit_rate": round(mean(result["checks"]["title_hit"] for result in results), 4),
        "section_hit_rate": round(mean(result["checks"]["section_hit"] for result in results), 4),
        "no_table_rate": round(mean(result["checks"]["no_table_ok"] for result in results), 4),
        "no_quote_rate": round(mean(result["checks"]["no_quote_ok"] for result in results), 4),
    }


def evaluate_with_ragas(
    results: list[dict[str, Any]],
    judge_profile_name: str,
):
    eligible = [result for result in results if result.get("ground_truth")]
    if not eligible:
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness
    except Exception as exc:
        return {"error": f"ragas import failed: {exc}"}

    dataset = Dataset.from_dict(
        {
            "question": [result["question"] for result in eligible],
            "answer": [result["answer"] for result in eligible],
            "contexts": [result["contexts"] for result in eligible],
            "ground_truth": [result["ground_truth"] for result in eligible],
            "reference": [result["ground_truth"] for result in eligible],
        }
    )

    try:
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=build_chat_model(judge_profile_name),
            embeddings=build_embedding_model(judge_profile_name),
        )
    except Exception as exc:
        return {"error": f"ragas evaluation failed: {exc}"}

    try:
        return ragas_result.to_pandas().to_dict(orient="records")
    except Exception:
        return str(ragas_result)


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--profile", default=get_default_profile_name())
    parser.add_argument("--judge-profile")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--with-ragas", action="store_true")
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()

    if args.list_profiles:
        for name in get_profile_names():
            profile = resolve_profile(name)
            print(f"{profile.name}: {profile.description}")
        return

    case_path = Path(args.cases)
    output_path = Path(args.output)
    profile = resolve_profile(args.profile)
    judge_profile_name = args.judge_profile or profile.name

    cases = load_cases(case_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    results = [evaluate_case(case, profile.name) for case in cases]
    summary = build_summary(results)
    payload = {
        "project_root": str(PROJECT_ROOT),
        "cases_path": str(case_path),
        "profile": profile.to_public_dict(),
        "summary": summary,
        "results": results,
    }

    if args.with_ragas:
        payload["ragas"] = evaluate_with_ragas(results, judge_profile_name)

    write_output(output_path, payload)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved harness results to {output_path}")


if __name__ == "__main__":
    main()
