from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
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


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases" / "smoke_cases.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RUNS_DIR / "latest_results.json"
DEFAULT_HISTORY_DIR = DEFAULT_RUNS_DIR / "history"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Cases JSON must be a list.")
    return cases


def make_iso_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_.") or "run"


def build_run_id(started: datetime, profile_name: str, run_label: str | None) -> str:
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    profile_slug = sanitize_label(profile_name)
    label_slug = sanitize_label(run_label) if run_label else None
    if label_slug:
        return f"{timestamp}__{profile_slug}__{label_slug}"
    return f"{timestamp}__{profile_slug}"


def build_history_output_path(run_id: str) -> Path:
    return DEFAULT_HISTORY_DIR / f"{run_id}.json"


def get_case_id(case: dict[str, Any]) -> str:
    return str(case.get("id") or case["question"])


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

    case_id = get_case_id(case)
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


def get_failed_checks(result: dict[str, Any]) -> list[str]:
    return [
        check_name
        for check_name, passed in result.get("checks", {}).items()
        if not passed
    ]


def print_case_progress(
    index: int,
    total: int,
    case: dict[str, Any],
    result: dict[str, Any],
) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    detail_parts = [f"{result['latency_ms']} ms"]
    failed_checks = get_failed_checks(result)
    if failed_checks:
        detail_parts.append(f"failed_checks={', '.join(failed_checks)}")
    if result.get("missing_parts"):
        detail_parts.append(f"missing={', '.join(result['missing_parts'])}")
    if result.get("forbidden_parts_found"):
        detail_parts.append(
            f"forbidden={', '.join(result['forbidden_parts_found'])}"
        )

    print(
        f"[{index}/{total}] {get_case_id(case)} ({case.get('category', 'general')}) -> "
        f"{status} | {' | '.join(detail_parts)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--profile", default=get_default_profile_name())
    parser.add_argument("--judge-profile")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-label")
    parser.add_argument("--quiet", action="store_true")
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

    started = datetime.now(timezone.utc)
    run_id = build_run_id(started, profile.name, args.run_label)
    history_path = build_history_output_path(run_id)

    cases = load_cases(case_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    if not args.quiet:
        print(
            f"Starting harness run {run_id} | profile={profile.name} | cases={len(cases)}",
            flush=True,
        )

    results = []
    for index, case in enumerate(cases, start=1):
        if not args.quiet:
            print(
                f"[{index}/{len(cases)}] Running {get_case_id(case)}...",
                flush=True,
            )
        result = evaluate_case(case, profile.name)
        results.append(result)
        if not args.quiet:
            print_case_progress(index, len(cases), case, result)

    summary = build_summary(results)
    finished = datetime.now(timezone.utc)
    payload = {
        "run_id": run_id,
        "run_label": args.run_label,
        "started_at": make_iso_timestamp(started),
        "finished_at": make_iso_timestamp(finished),
        "project_root": str(PROJECT_ROOT),
        "profile_name": profile.name,
        "cases_path": str(case_path),
        "case_count": len(cases),
        "with_ragas": args.with_ragas,
        "git_commit": get_git_commit(),
        "profile": profile.to_public_dict(),
        "summary": summary,
        "results": results,
    }

    if args.with_ragas:
        if not args.quiet:
            eligible_count = sum(1 for result in results if result.get("ground_truth"))
            print(
                f"Running RAGAS for {eligible_count} eligible cases with judge profile={judge_profile_name}",
                flush=True,
            )
        payload["ragas"] = evaluate_with_ragas(results, judge_profile_name)

    write_output(output_path, payload)
    write_output(history_path, payload)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved harness results to {output_path}")
    print(f"Saved harness history to {history_path}")


if __name__ == "__main__":
    main()
