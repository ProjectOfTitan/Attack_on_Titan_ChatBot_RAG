
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from uuid import uuid4

from back import get_ai_response
from mongoDB import insert_chat_log, update_feedback


st.set_page_config(page_title="TITAN_CHAT", page_icon="⚔️")

st.title("All About 진격의 거인")
st.caption("진격거에 관련된 모든것을 답해드립니다!")

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
        try:
            result = insert_chat_log(
                session_id=st.session_state.session_id,
                user_query=user_question,
                ai_response=ai_message,
                retrieved_context=retrieved_context,
            )
            log_id = str(result.inserted_id)
            st.session_state.last_log_id = log_id
            st.session_state.message_list[-1]["log_id"] = log_id
        except Exception as exc:
            st.warning(f"MongoDB 저장에 실패했습니다: {exc}")

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
                try:
                    update_feedback(log_id=log_id, feedback="like")
                    st.session_state.feedback_by_log_id[log_id] = "like"
                    st.rerun()
                except Exception as exc:
                    st.warning(f"피드백 저장에 실패했습니다: {exc}")
        with dislike_col:
            if st.button("싫어요", key=f"dislike_{log_id}"):
                try:
                    update_feedback(log_id=log_id, feedback="dislike")
                    st.session_state.feedback_by_log_id[log_id] = "dislike"
                    st.rerun()
                except Exception as exc:
                    st.warning(f"피드백 저장에 실패했습니다: {exc}")

