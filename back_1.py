from functools import lru_cache
from pathlib import Path

import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from dotenv import load_dotenv

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

from config import answer_examples

load_dotenv()



store = {}

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "aot_system_prompt.txt"

RELATIONSHIP_KEYWORDS = [
    "관계", "사이", "좋아해", "호감", "짝사랑", "연애", "결혼", "친구", "동료", 
    "유대", "감정", "심리", "커플", "러브라인", "호칭"
]

REPLACEMENTS = {
    "사람을 나타내는 표현": "진격의거인 애니메이션 안의 등장인물이나 사람들",
}

TABLE_ALLOWED_KEYWORDS = [
    "소개",
    "프로필",
    "정보",
    "설명",
    "요약",
    "정체",
    "인물",
    "캐릭터",
    "설정",
    "신체",
    "키",
    "나이",
    "생일",
    "출신",
    "소속",
    "계급",
    "가문",
    "종족",
    "능력",
    "전투력",
    "특징",
    "목록",
    "순위",
    "랭킹",
    "멤버",
    "구성원",
    "등장인물",
    "일람",
    "정리",
]
TABLE_ALLOWED_PATTERN = re.compile(r"(구성원|멤버|명단|리스트|계보)")

# 1. LLM이 생성할 질문 리스트를 위한 스키마 정의
class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def get_multiquery_retriever(base_retriever):
    """
    LLM을 사용하여 질문을 확장하고, 확장된 질문들로 검색을 수행하는 리트리버를 반환합니다.
    """
    llm = get_llm()
    
    # 진격의 거인 지식 검색에 최적화된 질문 확장 프롬프트
    query_expansion_prompt = PromptTemplate(
        input_variables=["question"],
        template="""당신은 '진격의 거인' 설정 및 인물 관계 전문가입니다.
사용자의 질문에 대해 가장 정확한 정보를 찾을 수 있도록 3개의 다양한 검색어(질문)를 생성하세요.

특히 '인물 간의 관계', '감정', '작중 사건'을 찾을 수 있도록 고유 명사를 포함하여 구체적으로 변환하세요.

사용자 질문: {question}

출력 형식:
1. 질문 1
2. 질문 2
3. 질문 3
"""
    )

    # MultiQueryRetriever 설정
    # parser_key는 기본적으로 'lines'를 사용하도록 설정되어 있습니다.
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=query_expansion_prompt
    )
    return mq_retriever

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print(store)
    return store[session_id]


@lru_cache(maxsize=1)
def get_embeddings():
    return UpstageEmbeddings(model='solar-embedding-1-large')

# 1. Chroma DB 객체만 캐싱합니다 (이 객체는 인자가 없어 에러가 나지 않습니다)
@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        collection_name='AoT', 
        persist_directory="./AoT", 
        embedding_function=get_embeddings()
    )


# 2. 리트리버 함수에서는 @lru_cache를 제거합니다. 
# 대신 위에서 캐싱된 vectorstore를 사용하므로 속도는 여전히 빠릅니다.
def get_retriever(search_filter: dict = None, k: int = 4):
    """
    필터링 조건과 검색 개수(k)를 받아 리트리버를 생성합니다.
    """
    database = Chroma(
        collection_name='AoT', 
        persist_directory="./AoT", 
        embedding_function=get_embeddings()
    )
    # search_kwargs에 k값을 전달하여 검색 결과 개수를 제어합니다.
    return database.as_retriever(search_kwargs={'k': k, 'filter': search_filter})
# 3. 히스토리 리트리버도 캐시 없이 호출되도록 합니다.
def get_history_retriever(search_filter: dict = None):
    llm = get_llm()
    retriever = get_retriever(search_filter)
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def get_llm(model='solar-pro2'):
    llm = ChatUpstage(model=model, temperature=0)
    return llm


def normalize_question(question: str) -> str:
    normalized = question
    for source, target in REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized

def _sanitize_metadata(metadata: dict) -> dict:
    if not metadata: return {}
    return {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) 
            for k, v in metadata.items()}

def retrieve_docs(question: str, session_id: str):
    """
    프리 필터링과 멀티 쿼리를 결합하여 문서를 검색합니다.
    """
    # 1. 관계 질문 여부 파악 및 프리 필터링 설정
    is_relational = any(keyword in question for keyword in RELATIONSHIP_KEYWORDS)
    search_filter = None
    if is_relational:
        search_filter = {
            "$and": [
                {"is_table": {"$eq": False}},
                {"is_quote": {"$eq": False}}
            ]
        }
    
    # 2. 기본 리트리버 설정 (k=4)
    # 멀티 쿼리가 질문을 3개로 늘리므로, 최종적으로 중복 제거 후 4~10개 사이의 결과가 나옵니다.
    base_retriever = get_retriever(search_filter, k=4)
    
    # 3. 멀티 쿼리 리트리버로 업그레이드
    final_retriever = get_multiquery_retriever(base_retriever)

    history = get_session_history(session_id)
    
    if len(history.messages) >= 2:
        # 히스토리(대화 흐름) 반영
        llm = get_llm()
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        h_retriever = create_history_aware_retriever(llm, final_retriever, contextualize_q_prompt)
        docs = h_retriever.invoke({"input": question, "chat_history": history.messages})
    else:
        # 단발성 질문
        docs = final_retriever.invoke(question)
        
    return docs


def get_answer_chain():
    llm = get_llm()
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
    system_prompt = get_system_prompt()
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain


def get_ai_response(user_message, session_id: str = "abc123"):
    normalized_question = normalize_question(user_message)
    
    # 문서 검색 (프리 필터링 적용됨)
    docs = retrieve_docs(normalized_question, session_id)
    retrieved_context = [
        {"text": doc.page_content, "metadata": _sanitize_metadata(doc.metadata)}
        for doc in docs
    ]

    history = get_session_history(session_id)
    answer_chain = get_answer_chain()
    
    inputs = {
        "input": normalized_question,
        "context": docs,
        "chat_history": history.messages,
    }
    
    # 스트리밍 답변 생성
    def _stream_answer():
        buffer = []
        for chunk in answer_chain.stream(inputs):
            buffer.append(chunk)
            yield chunk
        answer = "".join(buffer)
        history.add_user_message(user_message)
        history.add_ai_message(answer)

    return _stream_answer(), retrieved_context
