import streamlit as st
import tempfile
import os
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 기본 설정
st.set_page_config(page_title="RAG 챗봇", page_icon="🧩")
st.title("🧩 데이터 기반 RAG 챗봇")

# 2. 경로 상수 설정 (리포지토리 구조에 맞춤)
# 현재 파일(wiki.py)이 있는 폴더 기준으로 data/attack_on_Titan_Namu.jsonl을 찾습니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(CURRENT_DIR, "data", "attack_on_Titan_Namu.jsonl")
INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index_store")

# 3. 사이드바: API 키 입력만 받음 (파일 업로드 불필요)
with st.sidebar:
    st.header("설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.info("리포지토리 내장 데이터를 사용합니다.")

    # (선택사항) 강제로 다시 학습시키고 싶을 때 누르는 버튼
    if st.button("DB 재설정"):
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
            st.cache_resource.clear()
            st.rerun() # 앱 새로고침

# 3. RAG 핵심 로직 (캐싱 적용)
# @st.cache_resource는 벡터 DB 생성이 오래 걸리므로, 파일이 바뀌지 않으면 결과를 메모리에 저장해둡니다.
@st.cache_resource
def process_document(file_content):
    # Streamlit 업로드 파일은 바이너리 형태이므로 임시 파일로 저장 후 로드
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    try:
        documents = []
        # [핵심 수정] 파일을 한 줄씩 읽어서 파싱
        with open(tmp_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue # 빈 줄 건너뛰기
                
                try:
                    data = json.loads(line)
                    
                    # ⚠️ 중요: JSON에서 실제 '내용'이 들어있는 키(key) 이름을 맞춰야 합니다.
                    # 예: {"title": "...", "content": "본문내용..."} 이라면 "content"를 가져와야 함.
                    # 여기서는 'text', 'content', 'body' 중 하나를 자동으로 찾도록 했습니다.
                    text_content = data.get("content") or data.get("text") or data.get("body")
                    
                    # 만약 특정 키가 없고 전체를 다 쓰고 싶다면 아래 주석을 해제하세요.
                    # text_content = json.dumps(data, ensure_ascii=False)

                    if text_content:
                        # LangChain Document 객체 생성 (메타데이터도 같이 저장하면 좋음)
                        doc = Document(page_content=text_content, metadata=data)
                        documents.append(doc)
                except json.JSONDecodeError:
                    continue # 깨진 라인은 무시

        # 문서가 비어있으면 에러 처리
        if not documents:
            raise ValueError("JSONL 파일에서 유효한 텍스트를 추출하지 못했습니다. 키(Key) 이름을 확인하세요.")

        # # B. 텍스트 분할 (Chunking)
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,  # 1000자 단위
        #     chunk_overlap=100 # 문맥 끊김 방지
        # )
        # splits = text_splitter.split_documents(documents)

        # C. 임베딩 및 벡터 저장소 생성
        # 주의: 캐싱 함수 안에서는 외부 변수(api_key)를 직접 쓰기보다 파라미터로 받거나 내부 처리해야 함.
        # 여기서는 편의상 embeddings 객체 생성 시 키가 필요하므로, 
        # 실제로는 이 함수 호출 전에 키가 있는지 확인해야 안전함.
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        return vectorstore

    finally:
        # 임시 파일 삭제
        os.remove(tmp_file_path)

# 4. 메인 로직 실행
if uploaded_file and openai_api_key:
    # 파일이 업로드 되면 벡터 DB 생성 (또는 캐시된 것 사용)
    with st.spinner("데이터를 분석하고 있습니다..."):
        try:
            vector_store = process_document(uploaded_file.getvalue())
            st.success("문서 학습 완료! 질문해주세요.")
        except Exception as e:
            st.error(f"오류 발생: API Key를 확인하거나 파일 인코딩을 확인하세요.\n{e}")
            st.stop()
            
    # 5. 채팅 인터페이스
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retriever 설정
            retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # 관련 문서 3개 참조

            # LLM 설정
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

            # 프롬프트 엔지니어링
            system_prompt = (
                "당신은 업로드된 문서의 내용을 기반으로 답변하는 봇입니다. "
                "Context에 있는 내용만 사용하여 답변하고, 모르면 모른다고 하세요. "
                "\n\nContext:\n{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            # Chain 실행
            chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, chain)
            
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    # 초기 화면 안내
    st.info("좌측 사이드바에서 OpenAI API Key를 입력하고, 데이터 파일을 업로드해주세요.")