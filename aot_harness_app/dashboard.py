from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from runtime import PROJECT_ROOT


APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "runs"
LATEST_RESULTS_PATH = RUNS_DIR / "latest_results.json"
HISTORY_DIR = RUNS_DIR / "history"


st.set_page_config(page_title="AoT Harness Dashboard", page_icon="📊", layout="wide")
st.title("AoT Harness Dashboard")
st.caption("하네스 실행 결과를 누적 비교하고, 실패 원인을 빠르게 추적하기 위한 평가 전용 대시보드입니다.")


def make_iso_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0.0,
            "title_hit_rate": 0.0,
            "section_hit_rate": 0.0,
            "no_table_rate": 0.0,
            "no_quote_rate": 0.0,
        }

    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    failed = total - passed
    latency_values = [result.get("latency_ms", 0) for result in results]
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4),
        "avg_latency_ms": round(sum(latency_values) / total, 2),
        "title_hit_rate": round(sum(bool(result.get("checks", {}).get("title_hit")) for result in results) / total, 4),
        "section_hit_rate": round(sum(bool(result.get("checks", {}).get("section_hit")) for result in results) / total, 4),
        "no_table_rate": round(sum(bool(result.get("checks", {}).get("no_table_ok")) for result in results) / total, 4),
        "no_quote_rate": round(sum(bool(result.get("checks", {}).get("no_quote_ok")) for result in results) / total, 4),
    }


def summarize_ragas(ragas: Any) -> dict[str, Any]:
    if isinstance(ragas, dict):
        return ragas
    if not isinstance(ragas, list) or not ragas:
        return {}

    metric_values: dict[str, list[float]] = {}
    for row in ragas:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))

    return {
        key: round(sum(values) / len(values), 4)
        for key, values in metric_values.items()
        if values
    }


def format_run_label(run: dict[str, Any]) -> str:
    started = run["_started_dt"].astimezone().strftime("%Y-%m-%d %H:%M:%S")
    pass_rate = run["summary"].get("pass_rate", 0.0)
    return f"{started} | {run['profile_name']} | pass {pass_rate:.1%} | {run['run_id']}"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def coerce_run_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        results = []

    summary = payload.get("summary")
    if not isinstance(summary, dict) or "pass_rate" not in summary:
        summary = summarize_results(results)
    else:
        merged_summary = summarize_results(results)
        merged_summary.update(summary)
        summary = merged_summary

    started_dt = parse_timestamp(payload.get("started_at")) or datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )
    finished_dt = parse_timestamp(payload.get("finished_at")) or started_dt
    profile_dict = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    profile_name = payload.get("profile_name") or profile_dict.get("name") or "unknown"
    raw_case_count = payload.get("case_count")
    case_count = int(raw_case_count) if raw_case_count is not None else len(results)
    run_id = str(payload.get("run_id") or path.stem)
    ragas_summary = summarize_ragas(payload.get("ragas"))

    coerced = dict(payload)
    coerced.update(
        {
            "run_id": run_id,
            "profile_name": profile_name,
            "case_count": case_count,
            "summary": summary,
            "results": results,
            "with_ragas": bool(payload.get("with_ragas") or payload.get("ragas")),
            "ragas_summary": ragas_summary,
            "_path": str(path),
            "_started_dt": started_dt,
            "_finished_dt": finished_dt,
        }
    )
    coerced["_display_name"] = format_run_label(coerced)
    return coerced


@st.cache_data(show_spinner=False)
def load_runs() -> list[dict[str, Any]]:
    paths: list[Path] = []
    if HISTORY_DIR.exists():
        paths.extend(sorted(HISTORY_DIR.glob("*.json")))
    if LATEST_RESULTS_PATH.exists():
        paths.append(LATEST_RESULTS_PATH)

    runs_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        if payload is None:
            continue
        run = coerce_run_payload(path, payload)
        runs_by_id.setdefault(run["run_id"], run)

    return sorted(
        runs_by_id.values(),
        key=lambda run: run["_started_dt"],
    )


def build_runs_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {
            "run_id": run["run_id"],
            "profile_name": run["profile_name"],
            "started_at": pd.Timestamp(run["_started_dt"]),
            "finished_at": pd.Timestamp(run["_finished_dt"]),
            "case_count": run["case_count"],
            "pass_rate": run["summary"].get("pass_rate", 0.0),
            "passed": run["summary"].get("passed", 0),
            "failed": run["summary"].get("failed", 0),
            "avg_latency_ms": run["summary"].get("avg_latency_ms", 0.0),
            "title_hit_rate": run["summary"].get("title_hit_rate", 0.0),
            "no_table_rate": run["summary"].get("no_table_rate", 0.0),
            "no_quote_rate": run["summary"].get("no_quote_rate", 0.0),
            "git_commit": run.get("git_commit"),
            "display_name": run["_display_name"],
        }
        for key, value in run["ragas_summary"].items():
            if isinstance(value, (int, float)):
                row[f"ragas_{key}"] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("started_at")


def build_case_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        retrieval = result.get("retrieval", {})
        checks = result.get("checks", {})
        rows.append(
            {
                "case_id": result.get("case_id", ""),
                "category": result.get("category", "general"),
                "passed": bool(result.get("passed")),
                "latency_ms": result.get("latency_ms", 0),
                "context_count": retrieval.get("context_count", 0),
                "table_count": retrieval.get("table_count", 0),
                "quote_count": retrieval.get("quote_count", 0),
                "title_hit": bool(checks.get("title_hit")),
                "section_hit": bool(checks.get("section_hit")),
                "must_contain_ok": bool(checks.get("must_contain_ok")),
                "must_not_contain_ok": bool(checks.get("must_not_contain_ok")),
                "no_table_ok": bool(checks.get("no_table_ok")),
                "no_quote_ok": bool(checks.get("no_quote_ok")),
                "missing_parts": ", ".join(result.get("missing_parts", [])),
                "forbidden_parts_found": ", ".join(result.get("forbidden_parts_found", [])),
                "question": result.get("question", ""),
            }
        )
    return pd.DataFrame(rows)


def get_previous_run(runs: list[dict[str, Any]], current_run: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if run["profile_name"] == current_run["profile_name"]
        and run["_started_dt"] < current_run["_started_dt"]
    ]
    return candidates[-1] if candidates else None


def get_previous_case_result(
    runs: list[dict[str, Any]],
    current_run: dict[str, Any],
    case_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for run in reversed(runs):
        if run["profile_name"] != current_run["profile_name"]:
            continue
        if run["_started_dt"] >= current_run["_started_dt"]:
            continue
        for result in run["results"]:
            if result.get("case_id") == case_id:
                return run, result
    return None, None


def build_failure_frames(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    failed_results = [result for result in results if not result.get("passed")]

    category_counts = Counter(result.get("category", "general") for result in failed_results)
    category_df = pd.DataFrame(
        [{"category": key, "failed_cases": value} for key, value in category_counts.items()]
    )
    if not category_df.empty:
        category_df = category_df.sort_values("failed_cases", ascending=False)

    check_counts: Counter[str] = Counter()
    for result in failed_results:
        for check_name, passed in result.get("checks", {}).items():
            if not passed:
                check_counts[check_name] += 1
    check_df = pd.DataFrame(
        [{"check": key, "failed_cases": value} for key, value in check_counts.items()]
    )
    if not check_df.empty:
        check_df = check_df.sort_values("failed_cases", ascending=False)

    contamination_rows = [
        {
            "metric": "retrieval_table_contamination",
            "count": sum(result.get("retrieval", {}).get("table_count", 0) > 0 for result in results),
        },
        {
            "metric": "retrieval_quote_contamination",
            "count": sum(result.get("retrieval", {}).get("quote_count", 0) > 0 for result in results),
        },
        {
            "metric": "forbidden_table_failures",
            "count": sum(not result.get("checks", {}).get("no_table_ok", True) for result in results),
        },
        {
            "metric": "forbidden_quote_failures",
            "count": sum(not result.get("checks", {}).get("no_quote_ok", True) for result in results),
        },
    ]
    contamination_df = pd.DataFrame(contamination_rows)
    return category_df, check_df, contamination_df


def render_time_series(df: pd.DataFrame, y_field: str, title: str, percent_axis: bool = False) -> None:
    if df.empty:
        st.info("표시할 실행 이력이 없습니다.")
        return

    y_axis = alt.Y(y_field, title=title)
    if percent_axis:
        y_axis = alt.Y(y_field, title=title, axis=alt.Axis(format="%"))

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("started_at:T", title="Run Time"),
            y=y_axis,
            color=alt.Color("profile_name:N", title="Profile"),
            tooltip=[
                alt.Tooltip("display_name:N", title="Run"),
                alt.Tooltip("pass_rate:Q", title="Pass Rate", format=".1%"),
                alt.Tooltip("avg_latency_ms:Q", title="Avg Latency (ms)", format=".2f"),
                alt.Tooltip("started_at:T", title="Started"),
            ],
        )
        .properties(height=280, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def render_bar_chart(df: pd.DataFrame, x_field: str, y_field: str, title: str) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x_field, sort="-y", title=""),
            y=alt.Y(y_field, title="Count"),
            tooltip=list(df.columns),
        )
        .properties(height=280, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def render_run_metadata(run: dict[str, Any]) -> None:
    with st.expander("Run Metadata", expanded=False):
        st.json(
            {
                "run_id": run["run_id"],
                "run_label": run.get("run_label"),
                "started_at": run.get("started_at", make_iso_timestamp(run["_started_dt"])),
                "finished_at": run.get("finished_at", make_iso_timestamp(run["_finished_dt"])),
                "profile_name": run["profile_name"],
                "case_count": run["case_count"],
                "with_ragas": run["with_ragas"],
                "git_commit": run.get("git_commit"),
                "cases_path": run.get("cases_path"),
                "project_root": run.get("project_root", str(PROJECT_ROOT)),
                "source_file": run["_path"],
            }
        )


def render_case_result(result: dict[str, Any], heading: str) -> None:
    st.subheader(heading)
    st.write(f"Case ID: `{result.get('case_id', '')}`")
    st.write(f"Category: `{result.get('category', 'general')}`")
    st.write(f"Passed: `{result.get('passed', False)}`")
    st.write(f"Latency: `{result.get('latency_ms', 0)} ms`")
    st.write("Question")
    st.write(result.get("question", ""))
    st.write("Answer")
    st.write(result.get("answer", ""))
    if result.get("missing_parts"):
        st.write("Missing Parts")
        st.write(result["missing_parts"])
    if result.get("forbidden_parts_found"):
        st.write("Forbidden Parts Found")
        st.write(result["forbidden_parts_found"])
    st.write("Checks")
    st.json(result.get("checks", {}))
    st.write("Retrieval Summary")
    st.json(result.get("retrieval", {}))
    st.write("Retrieved Context")
    for idx, item in enumerate(result.get("retrieved_context", []), start=1):
        metadata = item.get("metadata", {})
        title = metadata.get("title", "제목 없음")
        section = metadata.get("section", "섹션 없음")
        with st.expander(f"{idx}. {title} / {section}", expanded=False):
            st.write(item.get("text", ""))
            st.json(metadata)


runs = load_runs()
if not runs:
    st.warning("아직 저장된 하네스 실행 결과가 없습니다.")
    st.code("/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_harness_app/harness.py --profile openrouter_gemma_free_existing_aot")
    st.stop()


if st.button("새로고침"):
    st.cache_data.clear()
    st.rerun()


runs_df = build_runs_frame(runs)
profile_options = sorted({run["profile_name"] for run in runs})
default_profile = runs[-1]["profile_name"]

with st.sidebar:
    st.subheader("Dashboard Filters")
    selected_profiles = st.multiselect(
        "Profiles",
        options=profile_options,
        default=[default_profile],
    )
    filtered_runs = [run for run in runs if run["profile_name"] in selected_profiles] if selected_profiles else runs
    if not filtered_runs:
        st.error("선택한 프로필에 해당하는 실행 결과가 없습니다.")
        st.stop()

    run_labels = {run["_display_name"]: run for run in filtered_runs}
    selected_run_label = st.selectbox(
        "Run",
        options=list(run_labels.keys()),
        index=len(run_labels) - 1,
    )
    selected_run = run_labels[selected_run_label]
    available_categories = sorted({result.get("category", "general") for result in selected_run["results"]})
    selected_categories = st.multiselect(
        "Categories",
        options=available_categories,
        default=available_categories,
    )
    failed_only = st.checkbox("Failed Only", value=True)


previous_run = get_previous_run(runs, selected_run)
summary = selected_run["summary"]

st.subheader("Overview")
metric_columns = st.columns(6)
pass_delta = None
latency_delta = None
if previous_run is not None:
    pass_delta = summary.get("pass_rate", 0.0) - previous_run["summary"].get("pass_rate", 0.0)
    latency_delta = summary.get("avg_latency_ms", 0.0) - previous_run["summary"].get("avg_latency_ms", 0.0)

metric_columns[0].metric(
    "Pass Rate",
    f"{summary.get('pass_rate', 0.0):.1%}",
    None if pass_delta is None else f"{pass_delta:+.1%}",
)
metric_columns[1].metric("Passed", int(summary.get("passed", 0)))
metric_columns[2].metric("Failed", int(summary.get("failed", 0)))
metric_columns[3].metric(
    "Avg Latency",
    f"{summary.get('avg_latency_ms', 0.0):.2f} ms",
    None if latency_delta is None else f"{latency_delta:+.2f} ms",
)
metric_columns[4].metric("Title Hit Rate", f"{summary.get('title_hit_rate', 0.0):.1%}")
metric_columns[5].metric("No Table Rate", f"{summary.get('no_table_rate', 0.0):.1%}")

extra_metric_columns = st.columns(3)
extra_metric_columns[0].metric("No Quote Rate", f"{summary.get('no_quote_rate', 0.0):.1%}")
extra_metric_columns[1].metric("Section Hit Rate", f"{summary.get('section_hit_rate', 0.0):.1%}")
extra_metric_columns[2].metric("Case Count", selected_run["case_count"])

if selected_run["ragas_summary"]:
    st.write("RAGAS Summary")
    ragas_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in selected_run["ragas_summary"].items()]
    )
    st.dataframe(ragas_df, use_container_width=True, hide_index=True)

render_run_metadata(selected_run)

st.subheader("History")
history_df = runs_df[runs_df["profile_name"].isin(selected_profiles)] if selected_profiles else runs_df
history_col1, history_col2 = st.columns(2)
with history_col1:
    render_time_series(history_df, "pass_rate:Q", "Pass Rate Trend", percent_axis=True)
with history_col2:
    render_time_series(history_df, "avg_latency_ms:Q", "Average Latency Trend")

with st.expander("Run History Table", expanded=False):
    display_history_df = history_df.copy()
    if not display_history_df.empty:
        display_history_df["started_at"] = display_history_df["started_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_history_df["finished_at"] = display_history_df["finished_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display_history_df, use_container_width=True, hide_index=True)

selected_results = [
    result
    for result in selected_run["results"]
    if result.get("category", "general") in selected_categories
]
if failed_only:
    selected_results = [result for result in selected_results if not result.get("passed")]

case_df = build_case_frame(selected_results)
category_df, check_df, contamination_df = build_failure_frames(selected_results)

st.subheader("Failure Analysis")
st.caption(
    f"현재 선택된 run에서 category 필터와 failed-only 조건을 적용한 "
    f"`{len(selected_results)}`개 케이스 기준 분석입니다."
)
failure_col1, failure_col2, failure_col3 = st.columns(3)
with failure_col1:
    render_bar_chart(category_df, "category:N", "failed_cases:Q", "Failed Cases by Category")
with failure_col2:
    render_bar_chart(check_df, "check:N", "failed_cases:Q", "Failed Checks")
with failure_col3:
    render_bar_chart(contamination_df, "metric:N", "count:Q", "Contamination")

st.subheader("Case Drilldown")
if case_df.empty:
    st.info("선택한 필터에 해당하는 케이스가 없습니다.")
    st.stop()

display_case_df = case_df.sort_values(["passed", "category", "case_id"], ascending=[True, True, True])
st.dataframe(display_case_df, use_container_width=True, hide_index=True)

case_options = display_case_df["case_id"].tolist()
selected_case_id = st.selectbox("Case ID", options=case_options)
selected_case = next(result for result in selected_results if result.get("case_id") == selected_case_id)

render_case_result(selected_case, "Selected Case")

st.subheader("Comparison")
previous_case_run, previous_case = get_previous_case_result(runs, selected_run, selected_case_id)
if previous_case is None or previous_case_run is None:
    st.info("같은 프로필에서 비교할 이전 케이스 실행이 없습니다.")
else:
    comparison_status = "동일"
    if selected_case.get("passed") and not previous_case.get("passed"):
        comparison_status = "개선"
    elif not selected_case.get("passed") and previous_case.get("passed"):
        comparison_status = "회귀"

    status_columns = st.columns(4)
    status_columns[0].metric("Status Change", comparison_status)
    status_columns[1].metric("Current Pass", str(selected_case.get("passed")))
    status_columns[2].metric("Previous Pass", str(previous_case.get("passed")))
    status_columns[3].metric(
        "Latency Delta",
        f"{selected_case.get('latency_ms', 0) - previous_case.get('latency_ms', 0):+d} ms",
    )

    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        st.caption(f"Current Run: {selected_run['_display_name']}")
        render_case_result(selected_case, "Current")
    with compare_col2:
        st.caption(f"Previous Run: {previous_case_run['_display_name']}")
        render_case_result(previous_case, "Previous")
