
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import threading
import time
from uuid import uuid4

import streamlit as st
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

from back import get_ai_response
from mongoDB import insert_chat_log, update_feedback


st.set_page_config(page_title="TITAN_CHAT", page_icon="⚔️")

st.title("All About 진격의 거인")
st.caption("진격거에 관련된 모든것을 답해드립니다!")

def _run_background(task, *args):
    thread = threading.Thread(target=task, args=args, daemon=True)
    thread.start()


def _insert_chat_log_task(log_id, session_id, user_query, ai_response, retrieved_context):
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


def _update_feedback_task(log_id, feedback, retries=5, delay=0.2):
    for _ in range(retries):
        result = update_feedback(log_id=log_id, feedback=feedback)
        if result.matched_count:
            return
        time.sleep(delay)
    print(f"피드백 저장 실패: {log_id}")


if 'message_list' not in st.session_state:
    st.session_state.message_list = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "feedback_by_log_id" not in st.session_state:
    st.session_state.feedback_by_log_id = {}

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="진격거에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response, retrieved_context = get_ai_response(
            user_question, st.session_state.session_id
        )
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            # print(st.session_state.message_list)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
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
            retrieved_context,
        )

if "last_log_id" in st.session_state:
    log_id = st.session_state.last_log_id
    existing_feedback = st.session_state.feedback_by_log_id.get(log_id)
    if existing_feedback:
        st.caption(f"피드백 저장됨: {existing_feedback}")
    else:
        st.write("답변이 도움이 되었나요?")
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
