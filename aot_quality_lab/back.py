from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterator

from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from config import answer_examples
from profiles import (
    build_chat_model,
    build_embedding_model,
    ensure_vectorstore_ready,
    resolve_profile,
)
from runtime import APP_ROOT, PROJECT_ROOT


PROMPT_PATH = APP_ROOT / "prompts" / "aot_system_prompt.txt"
REFUSAL_MESSAGE = "모르겠습니다. 진격의 거인과 관련된 질문만 답변할 수 있습니다."
FINAL_CONTEXT_K = 5
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")

RELATIONSHIP_KEYWORDS = [
    "관계",
    "사이",
    "좋아해",
    "호감",
    "짝사랑",
    "연애",
    "결혼",
    "친구",
    "동료",
    "유대",
    "감정",
    "심리",
    "커플",
    "러브라인",
    "호칭",
]

EPISODE_MEDIA_KEYWORDS = [
    "애니",
    "애니메이션",
    "원작",
    "만화",
    "몇 화",
    "몇화",
    "몇기",
    "몇 기",
    "기수",
    "ost",
    "ed",
    "op",
    "엔딩",
    "오프닝",
    "노래",
    "대사",
    "명대사",
    "내레이션",
    "누가말",
    "누가 말",
]

CHARACTER_LIST_KEYWORDS = [
    "멤버",
    "목록",
    "누구누구",
    "누구 누구",
    "전부",
    "정리해줘",
    "계승",
    "이름",
    "알려줘",
]

GUARDRAIL_KEYWORDS = [
    "날씨",
    "기온",
    "미세먼지",
    "비와",
    "수학",
    "방정식",
    "적분",
    "미분",
    "환율",
    "주가",
    "코드",
    "파이썬",
    "자바",
    "맛집",
    "식당",
    "번역",
    "영어로",
    "뉴스",
]

DOMAIN_HINT_KEYWORDS = [
    "진격",
    "거인",
    "에렌",
    "엘런",
    "엘렌",
    "미카사",
    "리바이",
    "아르민",
    "라이너",
    "베르톨트",
    "애니",
    "히스토리아",
    "크리스타",
    "유미르",
    "지크",
    "한지",
    "에르빈",
    "피크",
    "팔코",
    "조사병단",
    "헌병단",
    "주둔병단",
    "땅울림",
    "마레",
    "파라디",
    "시간시나",
    "리바이 반",
    "리바이반",
    "시조",
    "초대형",
    "갑옷",
    "여성형",
    "짐승",
    "턱",
    "차력",
    "전퇴",
]

STOPWORDS = {
    "진격",
    "거인",
    "진격의",
    "진격의거인",
    "애니",
    "애니메이션",
    "원작",
    "만화",
    "기준",
    "이거",
    "이거왜",
    "도대체",
    "알아",
    "알려줘",
    "정리해줘",
    "누구",
    "무엇",
    "뭐",
    "어때",
    "있어",
    "관련",
}

GENERAL_BAD_PATTERNS = [
    "2015년 영화",
    "극장판",
    "비판 및 논란",
    "결말 논란",
    "둘러보기",
    "가사",
]

REPLACEMENTS = {
    "사람을 나타내는 표현": "진격의거인 애니메이션 안의 등장인물이나 사람들",
    "진격의거인": "진격의 거인",
    "진격거": "진격의 거인",
    "리바이반": "리바이 반",
    "구 리바이반": "구 리바이 반",
    "캐니": "케니",
    "엘런": "에렌",
    "엘렌": "에렌",
    "수레 거인": "차력 거인",
    "카트 거인": "차력 거인",
    "공격 거인": "진격의 거인",
}

_store: dict[str, ChatMessageHistory] = {}


@dataclass
class ChatTurnResult:
    question: str
    normalized_question: str
    question_type: str
    retrieval_query: str
    answer: str
    retrieved_context: list[dict[str, Any]]
    search_filter: dict[str, Any] | None
    latency_ms: int
    session_id: str
    profile_name: str
    chat_model: str
    embedding_model: str
    collection_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def clear_session_history(session_id: str) -> None:
    _store.pop(session_id, None)


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in TOKEN_PATTERN.findall(text.lower()):
        token = match.strip()
        if len(token) < 2 or token in STOPWORDS or token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def classify_question(question: str) -> str:
    lowered = question.lower()
    has_domain_hint = _contains_keyword(lowered, DOMAIN_HINT_KEYWORDS)
    if _contains_keyword(lowered, GUARDRAIL_KEYWORDS) and not has_domain_hint:
        return "guardrail"
    if _contains_keyword(lowered, RELATIONSHIP_KEYWORDS):
        return "relationship"
    if _contains_keyword(lowered, EPISODE_MEDIA_KEYWORDS):
        return "episode_media"
    if _contains_keyword(lowered, CHARACTER_LIST_KEYWORDS):
        return "character_list"
    return "general_lore"


def build_retrieval_query(question: str, question_type: str) -> str:
    if question_type == "relationship":
        return f"{question} 인간관계 감정"

    if question_type == "episode_media":
        hints = []
        if _contains_keyword(question, ["ost", "엔딩", "오프닝", "노래", "음반", "ed", "op"]):
            hints.append("애니메이션 OST 엔딩 오프닝 음반")
        if _contains_keyword(question, ["몇 화", "몇화", "몇 기", "몇기", "기수", "애니", "원작", "만화"]):
            hints.append("애니메이션 원작 화수 기수")
        if _contains_keyword(question, ["내레이션", "명대사", "대사", "누가말", "누가 말"]):
            hints.append("애니메이션 내레이션 대사 어록")
        return " ".join([question, *hints]).strip()

    if question_type == "character_list":
        hints = ["이름", "멤버", "목록"]
        lowered = question.lower()
        if "리바이 반" in lowered:
            hints.extend(["리바이 반", "구 소속 병사"])
        if _contains_keyword(question, ["계승", "아홉 거인", "거인"]):
            hints.extend(["아홉 거인", "역대 계승자", "최종 계승자"])
        return " ".join([question, *hints]).strip()

    if _contains_keyword(question, ["조사병단", "헌병단", "주둔병단"]):
        return f"{question} 원작 TV 애니메이션 설정"

    return question


def _doc_key(doc) -> tuple[str, str, str]:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("title", "")),
        str(metadata.get("section", "")),
        str(metadata.get("chunk_index", "")),
    )


def dedup_docs(docs) -> list[Any]:
    deduped = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _structured_boost(text: str, metadata: dict[str, Any], question_type: str) -> float:
    score = 0.0
    title = str(metadata.get("title", "")).lower()
    section = str(metadata.get("section", "")).lower()
    if question_type == "character_list":
        if any(pattern in text for pattern in ["구 소속 병사", "소속 병사", "역대 계승자", "최종 계승자"]):
            score += 10.0
        if "리바이 반" in text or "리바이 반" in title or "리바이 반" in section:
            score += 12.0
        if metadata.get("is_table") is True:
            score += 4.0
    elif question_type == "episode_media":
        if any(pattern in title or pattern in section for pattern in ["애니메이션", "줄거리", "회차", "음반"]):
            score += 7.0
        if any(pattern in text for pattern in ["엔딩", "오프닝", "ost", "TV 애니메이션", "원작"]):
            score += 5.0
    elif question_type == "relationship":
        if "인간관계" in title or "인간관계" in section:
            score += 10.0
        if "작중 행적" in title or "작중 행적" in section:
            score += 3.0
    return score


def _contamination_penalty(text: str, metadata: dict[str, Any], question_type: str, question: str) -> float:
    penalty = 0.0
    title = str(metadata.get("title", "")).lower()
    section = str(metadata.get("section", "")).lower()
    combined = f"{title} {section} {text}"

    if question_type == "relationship":
        if metadata.get("is_table") is True:
            penalty -= 40.0
        if metadata.get("is_quote") is True:
            penalty -= 20.0

    if question_type == "episode_media":
        if _contains_keyword(question, ["ost", "엔딩", "오프닝", "노래", "음반", "ed", "op"]):
            if "비판 및 논란" in title or "2015년 영화" in title:
                penalty -= 20.0
        else:
            if "음반" in title or "가사" in section:
                penalty -= 12.0

    if question_type == "general_lore":
        if metadata.get("is_table") is True:
            penalty -= 6.0
        if metadata.get("is_quote") is True:
            penalty -= 3.0

    if question_type == "character_list":
        if "비판 및 논란" in combined or "둘러보기" in combined:
            penalty -= 22.0
    else:
        for pattern in GENERAL_BAD_PATTERNS:
            if pattern.lower() in combined:
                penalty -= 18.0

    return penalty


def score_doc(doc, *, question: str, question_type: str) -> float:
    metadata = doc.metadata or {}
    title = str(metadata.get("title", "")).lower()
    section = str(metadata.get("section", "")).lower()
    text = doc.page_content.lower()
    score = 0.0

    for token in tokenize(question):
        if token in title:
            score += 6.0
        if token in section:
            score += 3.0
        if token in text:
            score += 1.0

    if question.lower() in text:
        score += 6.0

    score += _structured_boost(text, metadata, question_type)
    score += _contamination_penalty(text, metadata, question_type, question)
    return score


def rerank_docs(docs, *, question: str, question_type: str):
    deduped = dedup_docs(docs)
    ranked = sorted(
        deduped,
        key=lambda doc: score_doc(doc, question=question, question_type=question_type),
        reverse=True,
    )
    return ranked


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def get_embeddings(profile_name: str):
    return build_embedding_model(profile_name)


@lru_cache(maxsize=8)
def get_vectorstore(profile_name: str) -> Chroma:
    profile = resolve_profile(profile_name)
    ensure_vectorstore_ready(profile.name)
    return Chroma(
        collection_name=profile.vectorstore.collection_name,
        persist_directory=profile.vectorstore.persist_directory,
        embedding_function=get_embeddings(profile.name),
    )


@lru_cache(maxsize=8)
def get_llm(profile_name: str):
    return build_chat_model(profile_name)


def get_retriever(
    profile_name: str,
    search_filter: dict[str, Any] | None = None,
    k: int | None = None,
):
    profile = resolve_profile(profile_name)
    return get_vectorstore(profile.name).as_retriever(
        search_kwargs={"k": k or profile.retriever_k, "filter": search_filter}
    )


def get_multiquery_retriever(profile_name: str, base_retriever):
    profile = resolve_profile(profile_name)
    output_lines = "\n".join(
        f"{index}. 질문 {index}" for index in range(1, profile.query_variant_count + 1)
    )
    query_expansion_prompt = PromptTemplate(
        input_variables=["question"],
        template=f"""당신은 '진격의 거인' 설정 및 인물 관계 전문가입니다.
사용자의 질문에 대해 가장 정확한 정보를 찾을 수 있도록 {profile.query_variant_count}개의 다양한 검색어(질문)를 생성하세요.

특히 '인물 간의 관계', '감정', '작중 사건'을 찾을 수 있도록 고유 명사를 포함하여 구체적으로 변환하세요.

사용자 질문: {{question}}

출력 형식:
{output_lines}
""",
    )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=get_llm(profile.name),
        prompt=query_expansion_prompt,
    )


def normalize_question(question: str) -> str:
    normalized = question
    for source, target in REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def build_search_filter(question_type: str) -> dict[str, Any] | None:
    if question_type != "relationship":
        return None
    return {
        "$and": [
            {"is_table": {"$eq": False}},
            {"is_quote": {"$eq": False}},
        ]
    }


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        key: (
            value
            if isinstance(value, (str, int, float, bool)) or value is None
            else str(value)
        )
        for key, value in metadata.items()
    }


def _serialize_docs(docs) -> list[dict[str, Any]]:
    serialized = []
    for doc in docs:
        serialized.append(
            {
                "text": doc.page_content,
                "metadata": _sanitize_metadata(doc.metadata),
            }
        )
    return serialized


def _needs_relationship_fallback(docs, question: str) -> bool:
    if len(docs) < 2:
        return True
    if not docs:
        return True
    return score_doc(docs[0], question=question, question_type="relationship") < 8.0


def retrieve_docs(question: str, session_id: str, profile_name: str | None = None):
    profile = resolve_profile(profile_name)
    normalized_question = normalize_question(question)
    question_type = classify_question(normalized_question)
    search_filter = build_search_filter(question_type)
    history = get_session_history(session_id)
    retrieval_query = build_retrieval_query(normalized_question, question_type)

    if question_type == "guardrail":
        return (
            profile,
            normalized_question,
            question_type,
            normalized_question,
            search_filter,
            [],
        )

    base_retriever = get_retriever(profile.name, search_filter=search_filter)
    final_retriever = (
        get_multiquery_retriever(profile.name, base_retriever)
        if profile.multi_query
        else base_retriever
    )

    if len(history.messages) >= 2:
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware = create_history_aware_retriever(
            get_llm(profile.name),
            final_retriever,
            contextualize_prompt,
        )
        docs = history_aware.invoke(
            {"input": retrieval_query, "chat_history": history.messages}
        )
    else:
        docs = final_retriever.invoke(retrieval_query)

    reranked_docs = rerank_docs(docs, question=normalized_question, question_type=question_type)

    if question_type == "relationship" and _needs_relationship_fallback(reranked_docs, normalized_question):
        relaxed_docs = get_retriever(profile.name, search_filter=None).invoke(retrieval_query)
        reranked_docs = rerank_docs(
            [*reranked_docs, *relaxed_docs],
            question=normalized_question,
            question_type=question_type,
        )

    final_docs = reranked_docs[:FINAL_CONTEXT_K]
    return (
        profile,
        normalized_question,
        question_type,
        retrieval_query,
        search_filter,
        final_docs,
    )


@lru_cache(maxsize=8)
def get_answer_chain(profile_name: str):
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_system_prompt()),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_stuff_documents_chain(get_llm(profile_name), qa_prompt)


def _prepare_turn(question: str, session_id: str, profile_name: str | None = None):
    profile, normalized_question, question_type, retrieval_query, search_filter, docs = retrieve_docs(
        question,
        session_id,
        profile_name=profile_name,
    )
    history = get_session_history(session_id)
    inputs = {
        "input": normalized_question,
        "context": docs,
        "chat_history": history.messages,
    }
    return (
        profile,
        normalized_question,
        question_type,
        retrieval_query,
        search_filter,
        docs,
        history,
        inputs,
    )


def answer_question(
    question: str,
    session_id: str = "aot_harness",
    commit_history: bool = True,
    profile_name: str | None = None,
) -> ChatTurnResult:
    started = time.perf_counter()
    (
        profile,
        normalized_question,
        question_type,
        retrieval_query,
        search_filter,
        docs,
        history,
        inputs,
    ) = _prepare_turn(
        question,
        session_id,
        profile_name=profile_name,
    )
    if question_type == "guardrail":
        answer = REFUSAL_MESSAGE
    else:
        answer = get_answer_chain(profile.name).invoke(inputs)
    if commit_history:
        history.add_user_message(question)
        history.add_ai_message(answer)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return ChatTurnResult(
        question=question,
        normalized_question=normalized_question,
        question_type=question_type,
        retrieval_query=retrieval_query,
        answer=answer,
        retrieved_context=_serialize_docs(docs),
        search_filter=search_filter,
        latency_ms=latency_ms,
        session_id=session_id,
        profile_name=profile.name,
        chat_model=profile.chat.model,
        embedding_model=profile.embedding.model,
        collection_name=profile.vectorstore.collection_name,
    )


def stream_question(
    question: str,
    session_id: str = "aot_harness",
    commit_history: bool = True,
    profile_name: str | None = None,
) -> tuple[Iterator[str], dict[str, Any]]:
    started = time.perf_counter()
    (
        profile,
        normalized_question,
        question_type,
        retrieval_query,
        search_filter,
        docs,
        history,
        inputs,
    ) = _prepare_turn(
        question,
        session_id,
        profile_name=profile_name,
    )
    turn_state = {
        "question": question,
        "normalized_question": normalized_question,
        "question_type": question_type,
        "retrieval_query": retrieval_query,
        "retrieved_context": _serialize_docs(docs),
        "search_filter": search_filter,
        "session_id": session_id,
        "answer": "",
        "latency_ms": None,
        "profile_name": profile.name,
        "chat_model": profile.chat.model,
        "embedding_model": profile.embedding.model,
        "collection_name": profile.vectorstore.collection_name,
    }

    def _stream() -> Iterator[str]:
        if question_type == "guardrail":
            turn_state["answer"] = REFUSAL_MESSAGE
            turn_state["latency_ms"] = int((time.perf_counter() - started) * 1000)
            if commit_history:
                history.add_user_message(question)
                history.add_ai_message(REFUSAL_MESSAGE)
            yield REFUSAL_MESSAGE
            return

        buffer: list[str] = []
        for chunk in get_answer_chain(profile.name).stream(inputs):
            buffer.append(chunk)
            yield chunk
        answer = "".join(buffer)
        if commit_history:
            history.add_user_message(question)
            history.add_ai_message(answer)
        turn_state["answer"] = answer
        turn_state["latency_ms"] = int((time.perf_counter() - started) * 1000)

    return _stream(), turn_state


def summarize_contexts(retrieved_context: list[dict[str, Any]]) -> dict[str, Any]:
    titles = []
    sections = []
    table_count = 0
    quote_count = 0

    for item in retrieved_context:
        metadata = item.get("metadata", {})
        title = str(metadata.get("title", "")).strip()
        section = str(metadata.get("section", "")).strip()
        if title:
            titles.append(title)
        if section:
            sections.append(section)
        if metadata.get("is_table") is True:
            table_count += 1
        if metadata.get("is_quote") is True:
            quote_count += 1

    return {
        "context_count": len(retrieved_context),
        "titles": titles,
        "sections": sections,
        "table_count": table_count,
        "quote_count": quote_count,
    }
