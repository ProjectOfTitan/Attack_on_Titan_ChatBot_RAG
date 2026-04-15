# AoT Quality Lab

기존 `aot_harness_app`를 보존한 채, strict eval과 retrieval 개선을 실험하기 위한 분리 워크스페이스입니다.

- `back.py`: guardrail 선차단, 질문 유형 분류, dedup + rerank, `8 -> 4` 컨텍스트 제한이 들어간 RAG 엔진
- `front.py`: 검색 컨텍스트와 question type, retrieval query를 함께 보는 Streamlit 채팅 UI
- `harness.py`: strict smoke case를 평가하는 오프라인 하네스
- `dashboard.py`: false positive와 실패 원인을 비교하는 대시보드
- `cases/smoke_cases.json`: stricter smoke case 셋

## 핵심 차이

- guardrail 질문은 retrieval 없이 바로 거절합니다.
- retrieval은 `candidate_k=8`로 가져오고, rerank 후 최종 `4`개만 answer chain에 넣습니다.
- smoke case는 `forbidden_titles`, `expected_titles_mode`, `required_pairs`를 지원합니다.

## 실행

```bash
/mnt/e/one_piece/venv/bin/streamlit run /mnt/e/one_piece/aot_quality_lab/front.py
```

```bash
/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_quality_lab/harness.py --profile upstage_prod
```

```bash
/mnt/e/one_piece/venv/bin/streamlit run /mnt/e/one_piece/aot_quality_lab/dashboard.py
```

## 결과 파일

- 최신 실행: `aot_quality_lab/runs/latest_results.json`
- 실행 이력: `aot_quality_lab/runs/history/<timestamp>__<profile>.json`

## 메모

- 이 폴더는 `/mnt/e/one_piece/prompts`와 `/mnt/e/one_piece/config.py`를 복사한 로컬 사본을 사용합니다.
- 원본 `aot_harness_app`은 그대로 유지됩니다.
