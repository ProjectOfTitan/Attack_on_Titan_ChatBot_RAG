# 프로젝트 개요

진격의 거인 나무위키 카테고리 페이지를 시작점으로 크롤링해 RAG에 적합한 구조화 텍스트(JSONL)를 만드는 프로젝트입니다.  
카테고리 영역의 링크만 따라가며, 불필요한 UI/광고/이미지 요소를 제거해 환각을 줄이는 것을 목표로 합니다.  
추가로 Streamlit 앱으로 데이터를 업로드해 열거(목록 확인) 및 질의응답을 진행할 수 있습니다.

## 주요 기능
- 카테고리 링크 제한 추적: `#category-문서`, `#category-분류` 내부 링크만 따라감
- 해당 컨테이너가 없으면 해당 페이지는 크롤링하되 링크는 따라가지 않음
- 제목/문단/리스트/인용문/표/토글 요소를 구조화해 추출
- 이미지 제거, 링크는 텍스트만 남김(URL payload 제거)
- 광고/TOC/카테고리 박스 등 불필요한 DOM 필터링
- 섹션 단위로 청크 분할(섹션이 바뀌면 overlap 중단)
- 표/인용문은 단독 청크로 분리하고 마커 부여
- 500페이지 단위로 JSONL 파일 분할 저장

## 출력 형식(JSONL)
각 줄은 아래 구조의 JSON 객체입니다.
```json
{
  "title": "페이지 제목",
  "section": "섹션 제목",
  "chunk_index": "12",
  "text": "청크 내용..."
}
```

## 크롤러 실행
현재 필터링 버전은 `namu_category_crawler_filtered.py` 입니다.
```bash
python namu_category_crawler_filtered.py \
  --max-pages 500 \
  --output data/attack_on_Titan_Namu.jsonl
```
출력은 `data/attack_on_Titan_Namu_part1.jsonl`, `data/attack_on_Titan_Namu_part2.jsonl` 형태로 분할됩니다.

## Streamlit 앱
`front.py, back.py`는 RAG 질의응답을 수행하는 간단한 UI입니다.  
데이터를 업로드해 문서를 열거(목록 확인)하고 질문/응답 흐름을 검증하는 용도로 사용합니다.

## 기본 옵션
- `--chunk-size 1000`
- `--chunk-overlap 90` (같은 섹션 내에서만 overlap)
- `--min-block-chars 30`
- `--min-page-chars 300`
- `--min-paragraphs 1`

## RAG 품질 팁
