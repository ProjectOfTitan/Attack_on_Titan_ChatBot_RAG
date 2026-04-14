from __future__ import annotations

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
from runtime import PROJECT_ROOT


PROMPT_PATH = PROJECT_ROOT / "prompts" / "aot_system_prompt.txt"

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

REPLACEMENTS = {
    "사람을 나타내는 표현": "진격의거인 애니메이션 안의 등장인물이나 사람들",
}

_store: dict[str, ChatMessageHistory] = {}


@dataclass
class ChatTurnResult:
    question: str
    normalized_question: str
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


def build_search_filter(question: str) -> dict[str, Any] | None:
    is_relational = any(keyword in question for keyword in RELATIONSHIP_KEYWORDS)
    if not is_relational:
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


def retrieve_docs(question: str, session_id: str, profile_name: str | None = None):
    profile = resolve_profile(profile_name)
    normalized_question = normalize_question(question)
    search_filter = build_search_filter(normalized_question)
    history = get_session_history(session_id)

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
            {"input": normalized_question, "chat_history": history.messages}
        )
    else:
        docs = final_retriever.invoke(normalized_question)

    return profile, normalized_question, search_filter, docs


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
    profile, normalized_question, search_filter, docs = retrieve_docs(
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
    return profile, normalized_question, search_filter, docs, history, inputs


def answer_question(
    question: str,
    session_id: str = "aot_harness",
    commit_history: bool = True,
    profile_name: str | None = None,
) -> ChatTurnResult:
    started = time.perf_counter()
    profile, normalized_question, search_filter, docs, history, inputs = _prepare_turn(
        question,
        session_id,
        profile_name=profile_name,
    )
    answer = get_answer_chain(profile.name).invoke(inputs)
    if commit_history:
        history.add_user_message(question)
        history.add_ai_message(answer)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return ChatTurnResult(
        question=question,
        normalized_question=normalized_question,
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
    profile, normalized_question, search_filter, docs, history, inputs = _prepare_turn(
        question,
        session_id,
        profile_name=profile_name,
    )
    turn_state = {
        "question": question,
        "normalized_question": normalized_question,
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
