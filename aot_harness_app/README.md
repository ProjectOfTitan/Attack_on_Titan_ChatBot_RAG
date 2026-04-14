# AoT Harness App

기존 `front.py`, `back_1.py`, `ragas_batch.py`를 기준으로 다시 묶은 폴더입니다.

- `back.py`: 프로필 기반으로 생성 모델, 임베딩, 벡터스토어를 바꿔가며 같은 RAG 파이프라인을 실행합니다.
- `front.py`: Streamlit 채팅 UI입니다. 답변과 함께 검색된 컨텍스트를 바로 확인할 수 있습니다.
- `harness.py`: `back.py`를 직접 호출하는 오프라인 하네스입니다. 즉 프론트와 같은 프로필을 그대로 평가합니다.
- `dashboard.py`: 누적된 하네스 결과를 시각화하고 실패 케이스를 분석하는 Streamlit 대시보드입니다.
- `build_vectorstore.py`: 임베딩 프로필에 맞춰 새 Chroma 인덱스를 빌드합니다.

## 내장 프로필

- `upstage_prod`: 기존 서비스와 동일한 유료 Upstage 경로
- `openrouter_gemma_free_existing_aot`: 생성만 OpenRouter 무료 Gemma로 바꾸고, 기존 `AoT` 인덱스는 그대로 사용
- `openrouter_gemma_free_local_e5`: 생성은 OpenRouter 무료 Gemma, 임베딩은 로컬 `multilingual-e5`로 바꿔 새 인덱스를 사용

프로필은 UI에서 고를 수 있고, CLI에서는 `--profile`로 고를 수 있습니다.

## 환경 변수

최소한 아래 정도는 준비하는 편이 좋습니다.

```env
OPENROUTER_API_KEY=...
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_X_TITLE=AoT Harness
```

프로필을 코드 수정 없이 세밀하게 덮어쓰고 싶으면 아래도 사용할 수 있습니다.

```env
AOT_PROFILE=openrouter_gemma_free_existing_aot
AOT_CHAT_MODEL=google/gemma-4-31b-it:free
AOT_CHAT_PROVIDER=openai_compatible
AOT_CHAT_BASE_URL=https://openrouter.ai/api/v1
AOT_CHAT_API_KEY_ENV=OPENROUTER_API_KEY
```

## 실행

```bash
/mnt/e/one_piece/venv/bin/streamlit run /mnt/e/one_piece/aot_harness_app/front.py
```

```bash
/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_harness_app/harness.py --profile openrouter_gemma_free_existing_aot
```

대시보드를 보려면:

```bash
/mnt/e/one_piece/venv/bin/streamlit run /mnt/e/one_piece/aot_harness_app/dashboard.py
```

RAGAS까지 함께 돌리려면:

```bash
/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_harness_app/harness.py --profile openrouter_gemma_free_existing_aot --with-ragas
```

결과 JSON은 기본적으로 아래 두 곳에 저장됩니다.

- 최신 실행: `aot_harness_app/runs/latest_results.json`
- 실행 이력: `aot_harness_app/runs/history/<timestamp>__<profile>.json`

`--run-label`을 주면 run id에 라벨이 함께 반영됩니다.

```bash
/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_harness_app/harness.py --profile openrouter_gemma_free_existing_aot --run-label prompt-v2
```

대시보드에서는 다음을 볼 수 있습니다.

- 상단 KPI: pass rate, latency, title hit, table/quote 관련 비율
- 실행 이력: 프로필별 pass rate / latency 추이
- 실패 원인 분석: category별 실패 수, check별 실패 수, contamination 수치
- 케이스 드릴다운: 질문, 답변, 누락 키워드, retrieval summary, retrieved context
- 이전 run과 비교: 동일 case_id 기준 pass/fail 변화와 답변 비교

## 완전 무료 경로

기존 `AoT` 인덱스는 Upstage 임베딩으로 만들어져 있습니다. 생성 모델만 무료로 바꾸려면 `openrouter_gemma_free_existing_aot`를 쓰면 됩니다.

임베딩까지 무료로 가려면 새 인덱스를 먼저 만들어야 합니다.

```bash
/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_harness_app/build_vectorstore.py --profile openrouter_gemma_free_local_e5
```

그 다음:

```bash
/mnt/e/one_piece/venv/bin/streamlit run /mnt/e/one_piece/aot_harness_app/front.py
```

또는:

```bash
/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_harness_app/harness.py --profile openrouter_gemma_free_local_e5
```
