# 프로젝트 개요

진격의 거인(Attack on Titan) 지식을 대상으로 RAG 파이프라인을 구성해 질문에 답하는 챗봇 프로젝트입니다.
나무위키 카테고리 기반 크롤링 → 구조화(JSONL) → Chroma 벡터 저장 → Streamlit 챗 UI → MongoDB 로그 → RAGAS 평가까지의 흐름을 포함합니다.

## 핵심 기능
- 나무위키 카테고리 기반 크롤링 및 텍스트 구조화(JSONL)
- Chroma 벡터 스토어 기반 검색 + LLM 응답
- Streamlit 챗봇 UI
- 대화 로그/컨텍스트/피드백 MongoDB 저장
- RAGAS 평가 배치 실행
- LangGraph 버전의 그래프형 파이프라인(옵션)

## 구성 요소
- 크롤러: `namu_category_crawler_filtered.py`
- 벡터 저장소: `attackTitan/` (Chroma persist)
- 백엔드(RAG): `back.py`
- 프론트(Streamlit): `front.py`
- MongoDB 스키마/로그: `mongoDB.py`
- RAGAS 배치 평가: `ragas_batch.py`, `ragas_eval_from_json.py`
- LangGraph 실험 코드: `langgraph/`

## 빠른 시작

### 1) 가상환경 및 설치
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) 환경 변수 설정
프로젝트 루트에 `.env` 파일을 만들고 아래 값을 설정합니다.
```env
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=mongodb+srv://<id>:<pw>@attackoftitan.n8lazb9.mongodb.net/?appName=AttackofTitan
```

### 3) 챗봇 실행
```bash
streamlit run front.py
```

## 데이터 준비(선택)
크롤러 실행으로 JSONL 데이터를 생성할 수 있습니다.
```bash
python namu_category_crawler_filtered.py \
  --max-pages 200 \
  --output data/attack_on_Titan_Namu.jsonl
```

## RAGAS 평가

### 1) 배치 생성 + 평가
`ragas_batch.py`에 질문을 넣고 실행하면 답변을 생성하고 `ragas_answers.json`에 저장합니다.
```bash
python ragas_batch.py
```

### 2) 저장된 JSON으로 재평가
```bash
python ragas_eval_from_json.py --input ragas_answers.json
```

`ground_truth`가 있으면 다음처럼 추가 평가가 가능합니다.
```bash
python ragas_eval_from_json.py --input ragas_answers.json --ground-truths ground_truths.json
```

## MongoDB 로그 스키마
저장되는 로그는 다음 필드를 포함합니다.
- `session_id`: 사용자 세션 구분
- `user_query`: 사용자 질문
- `ai_response`: 모델 답변
- `retrieved_context`: 검색된 문서 조각(텍스트 + 메타데이터)
- `feedback`: (선택) 좋아요/싫어요
- `timestamp`: 대화 시간

## LangGraph(옵션)
`langgraph/` 폴더에 그래프 기반 파이프라인 예시가 있습니다.
```bash
python langgraph/run_graph.py --question "에렌은 왜 땅울림을 했어?"
```

## 히스토리
- 2026-01-05: RAG 파이프라인 최적화(검색 1회화, 룰 기반 치환, retriever 캐시) 및 로그 저장 비동기화 적용
- 2026-01-05: RAGAS 평가 스크립트 추가 및 JSON 저장/재평가 흐름 정리
- 2026-01-07: data의 section부분이 누락되는 것을 확인. 크롤링 부분 재점검 필요 느낌
- 2026-01-08: section 누락 부분 해결 완료 -> if문에 의해 잘못 파싱되는 문제가 있었음.
- 2026-01-09: upstage embedding model + Solar2 model 적용 (이전엔 openai api였음)
- 2026-01-09: AoT -> 새롭게 파싱해서 새로운 vectordb 생성(완)