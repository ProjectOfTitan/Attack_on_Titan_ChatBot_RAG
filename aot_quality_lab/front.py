from __future__ import annotations

import threading
import time
from uuid import uuid4

import streamlit as st
from bson import ObjectId

from back import stream_question, summarize_contexts
from profiles import (
    get_default_profile_name,
    get_profile_names,
    resolve_profile,
    vectorstore_exists,
)

try:
    from mongoDB import insert_chat_log, update_feedback
except Exception:
    insert_chat_log = None
    update_feedback = None


st.set_page_config(page_title="AoT Quality Lab Chat", page_icon="⚔️", layout="wide")

st.title("AoT Quality Lab Chat")
st.caption("strict eval과 retrieval 개선을 실험하는 Quality Lab UI입니다.")


def _run_background(task, *args) -> None:
    thread = threading.Thread(target=task, args=args, daemon=True)
    thread.start()


def _insert_chat_log_task(log_id, session_id, user_query, ai_response, retrieved_context):
    if insert_chat_log is None:
        return
    try:
        insert_chat_log(
            log_id=log_id,
            session_id=session_id,
            user_query=user_query,
            ai_response=ai_response,
            retrieved_context=retrieved_context,
        )
    except Exception as exc:
        print(f"MongoDB 저장 실패: {exc}")


def _update_feedback_task(log_id, feedback, retries: int = 5, delay: float = 0.2):
    if update_feedback is None:
        return
    for _ in range(retries):
        result = update_feedback(log_id=log_id, feedback=feedback)
        if result.matched_count:
            return
        time.sleep(delay)
    print(f"피드백 저장 실패: {log_id}")


if "message_list" not in st.session_state:
    st.session_state.message_list = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "feedback_by_log_id" not in st.session_state:
    st.session_state.feedback_by_log_id = {}
if "last_turn" not in st.session_state:
    st.session_state.last_turn = None
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = get_default_profile_name()


with st.sidebar:
    st.subheader("Pipeline Profile")
    profile_names = get_profile_names()
    current_profile = st.session_state.selected_profile
    selected_profile = st.selectbox(
        "실행 프로필",
        options=profile_names,
        index=profile_names.index(current_profile)
        if current_profile in profile_names
        else 0,
    )
    if selected_profile != st.session_state.selected_profile:
        st.session_state.selected_profile = selected_profile
        st.session_state.session_id = str(uuid4())
        st.session_state.message_list = []
        st.session_state.last_turn = None
        st.session_state.feedback_by_log_id = {}
        st.session_state.pop("last_log_id", None)

    profile = resolve_profile(st.session_state.selected_profile)
    vectorstore_ready = vectorstore_exists(profile.name)

    st.caption(profile.description)
    st.write(f"Chat: `{profile.chat.provider}` / `{profile.chat.model}`")
    st.write(f"Embed: `{profile.embedding.provider}` / `{profile.embedding.model}`")
    st.write(f"Collection: `{profile.vectorstore.collection_name}`")
    st.write(f"Session ID: `{st.session_state.session_id}`")
    if vectorstore_ready:
        st.caption("벡터스토어 준비됨")
    else:
        st.error("이 프로필의 벡터스토어가 아직 없습니다.")
        st.code(
            f"/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_quality_lab/build_vectorstore.py --profile {profile.name}"
        )

    if st.session_state.last_turn:
        last_turn = st.session_state.last_turn
        summary = summarize_contexts(last_turn["retrieved_context"])
        st.write(f"Contexts: {summary['context_count']}")
        st.write(f"Table Chunks: {summary['table_count']}")
        st.write(f"Quote Chunks: {summary['quote_count']}")
        if last_turn.get("search_filter"):
            st.caption("관계 질문 필터가 적용되었습니다.")
    else:
        st.caption("아직 실행된 질의가 없습니다.")


chat_col, debug_col = st.columns([1.4, 1.0], gap="large")

with chat_col:
    for message in st.session_state.message_list:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question := st.chat_input(
        "진격의 거인에 대해 질문해보세요.",
        disabled=not vectorstore_ready,
    ):
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})

        with st.spinner("답변을 생성하는 중입니다."):
            answer_stream, turn_state = stream_question(
                user_question,
                st.session_state.session_id,
                profile_name=st.session_state.selected_profile,
            )
            with st.chat_message("ai"):
                ai_message = st.write_stream(answer_stream)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
            turn_state["answer"] = ai_message
            st.session_state.last_turn = turn_state

        log_id = ObjectId()
        log_id_str = str(log_id)
        st.session_state.last_log_id = log_id_str
        st.session_state.message_list[-1]["log_id"] = log_id_str
        _run_background(
            _insert_chat_log_task,
            log_id,
            st.session_state.session_id,
            user_question,
            ai_message,
            turn_state["retrieved_context"],
        )

    if "last_log_id" in st.session_state:
        log_id = st.session_state.last_log_id
        existing_feedback = st.session_state.feedback_by_log_id.get(log_id)
        if existing_feedback:
            st.caption(f"피드백 저장됨: {existing_feedback}")
        else:
            like_col, dislike_col = st.columns(2)
            with like_col:
                if st.button("좋아요", key=f"like_{log_id}"):
                    st.session_state.feedback_by_log_id[log_id] = "like"
                    _run_background(_update_feedback_task, log_id, "like")
                    st.rerun()
            with dislike_col:
                if st.button("싫어요", key=f"dislike_{log_id}"):
                    st.session_state.feedback_by_log_id[log_id] = "dislike"
                    _run_background(_update_feedback_task, log_id, "dislike")
                    st.rerun()


with debug_col:
    st.subheader("Retrieved Context")
    last_turn = st.session_state.last_turn
    if not last_turn:
        st.info("질문을 한 번 실행하면 검색된 문서 조각이 여기에 표시됩니다.")
    else:
        if last_turn.get("latency_ms") is not None:
            st.caption(f"Latency: {last_turn['latency_ms']} ms")
        st.caption(f"Profile: {last_turn['profile_name']}")
        st.caption(f"Chat Model: {last_turn['chat_model']}")
        st.caption(f"Embedding Model: {last_turn['embedding_model']}")
        st.caption(f"Question Type: {last_turn['question_type']}")
        st.caption(f"Normalized Question: {last_turn['normalized_question']}")
        st.caption(f"Retrieval Query: {last_turn['retrieval_query']}")
        for idx, item in enumerate(last_turn["retrieved_context"], start=1):
            metadata = item.get("metadata", {})
            title = metadata.get("title", "제목 없음")
            section = metadata.get("section", "섹션 없음")
            header = f"{idx}. {title} / {section}"
            with st.expander(header, expanded=False):
                st.write(item.get("text", ""))
                st.json(metadata)
