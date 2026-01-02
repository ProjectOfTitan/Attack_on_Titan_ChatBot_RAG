__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv
from back import get_ai_response


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

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            # print(st.session_state.message_list)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})

    feedback_key = f"feedback_{len(st.session_state.message_list)}"
    
    col1, col2, _ = st.columns([0.1, 0.1, 0.8])
    with col1:
        if st.button("👍", key=f"up_{feedback_key}"):
            st.success("피드백 감사합니다!")
            # save_feedback_to_mongodb(user_question, ai_message, "good") # 나중에 구현할 함수
            
    with col2:
        if st.button("👎", key=f"down_{feedback_key}"):
            st.error("피드백 감사합니다!")
            # save_feedback_to_mongodb(user_question, ai_message, "bad")