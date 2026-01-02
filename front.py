__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv
from back import get_ai_response
from mongo import save_chat_log


st.set_page_config(page_title="TITAN_CHAT", page_icon="⚔️")

st.title("All About 진격의 거인")
st.caption("진격거에 관련된 모든것을 답해드립니다!")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="진격거에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
        with st.spinner("답변을 생성하는 중입니다"):
            full_response = ""
            retrieved_docs = []
            
            response_container = st.empty()
            for chunk in get_ai_response(user_question):
                if "context" in chunk:
                    retrieved_docs = [doc.page_content for doc in chunk["context"]]
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_container.write(full_response)
            
            st.session_state.message_list.append({"role": "ai", "content": full_response})
            
            # 데이터를 세션에 명확히 저장
            st.session_state.last_query = user_question
            st.session_state.last_response = full_response
            st.session_state.last_context = retrieved_docs
            # 답변이 완료되었음을 알리는 플래그
            st.session_state.show_feedback = True

# --- 이 부분이 블록 밖으로 나와야 합니다 ---
if st.session_state.get("show_feedback"):
    feedback_key = f"feedback_{len(st.session_state.message_list)}"
    
    col1, col2, _ = st.columns([0.1, 0.1, 0.8])
    with col1:
        if st.button("👍", key=f"up_{feedback_key}"):
            res = save_chat_log(
                st.session_state.last_query,
                st.session_state.last_response,
                st.session_state.last_context,
                "good"
            )
            if res:
                st.success("피드백이 DB에 저장되었습니다!")
                st.session_state.show_feedback = False # 중복 저장 방지
    with col2:
        if st.button("👎", key=f"down_{feedback_key}"):
            res = save_chat_log(
                st.session_state.last_query,
                st.session_state.last_response,
                st.session_state.last_context,
                "bad"
            )
            if res:
                st.error("피드백이 기록되었습니다.")
                st.session_state.show_feedback = False