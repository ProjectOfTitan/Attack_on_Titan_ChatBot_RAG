from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

from config import answer_examples

load_dotenv()



store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print(store)
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-small')
    # embedding = UpstageEmbeddings(model='solar-embedding-1-large')
    # index_name = 'law-table-index'
    # database = Chroma(collection_name='chroma-inu-new', persist_directory="./chroma_inu-new", embedding_function=embedding) # 학칙만
    database = Chroma(collection_name='attackTitan', persist_directory="./attackTitan", embedding_function=embedding) #학칙+장학금
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-5-mini'):
    llm = ChatOpenAI(model=model)
    # llm = Ollama(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 진격의거인 애니메이션 안의 등장인물이나 사람들"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def _sanitize_metadata(metadata: dict) -> dict:
    if not metadata:
        return {}
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def get_rag_chain():
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
    system_prompt = (
    """당신은 애니메이션 및 원작 만화 『진격의 거인』(Attack on Titan)의 설정, 스토리, 인물, 세계관에 대해
    정확하고 깊이 있는 지식을 가진 전문가입니다.
    사용자의 질문에 대해 오직 『진격의 거인』 작품 내부 정보에 기반하여서만 답변해야 합니다.

    [중요 규칙 1]
    사용자의 질문이 『진격의 거인』과 직접적으로 관련되지 않은 경우,
    추측하거나 일반 상식으로 답변하지 말고 반드시 다음과 같이 답변하십시오:
    "모르겠습니다. 진격의 거인과 관련된 질문만 답변할 수 있습니다."

    [중요 규칙 2]
    사용자가 인물 이름을 번역 차이, 음역 차이, 오타, 별칭 등으로 질문하더라도
    아래의 대응 관계를 자동으로 동일 인물로 인식해야 합니다.
    단, 답변에서는 반드시 한국 공식 번역 기준의 정식 이름을 사용하십시오.

    - 에렌 예거: 에렌, 엘런, 엘렌, 엘런 예거, 엘렌 예거, Eren, Yeager, Jaeger
    - 미카사 아커만: 미카사, 미카사 아카만, 아커만 미카사, Mikasa
    - 리바이 아커만: 리바이, 리바이 병장, Levi
    - 아르민 알레르토: 아르민, 알민, Armin
    - 라이너 브라운: 라이너, Reiner
    - 베르톨트 후버: 베르톨트, 베르톨트 후버, 베르톨트 훼버, Bertolt, Bertholdt
    - 애니 레온하트: 애니, Annie
    - 히스토리아 레이스: 히스토리아, 크리스타, Historia, Krista
    - 지크 예거: 지크, 지크 예거, Zeke
    - 한지 조에: 한지, 한지 조에, Hange, 한지 조에
    - 에르빈 스미스: 에르빈, Erwin
    - 피크 핑거: 피크, Pieck
    - 액커맨, 아커맨, 액커만, 아커만은 다 동일하게 아커만으로 인식하십시오.

    [중요 규칙 3]
    인물, 사건, 설정에 대해 답변할 때는
    - 작중 시점
    - 애니메이션 / 만화 기준 여부
    - 명확히 밝혀진 설정과 해석의 영역을 구분
    하여 서술하십시오.

    [중요 규칙 4]
    확실하지 않거나 작중에서 명시되지 않은 내용은
    사실처럼 단정하지 말고 "작중에서 명확히 밝혀지지 않았다"고 답변하십시오.

    아래는 참고용 컨텍스트입니다.
    \n\n{context}
    """
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain


def get_retrieved_context(user_message: str, session_id: str):
    dictionary_chain = get_dictionary_chain()
    normalized_question = dictionary_chain.invoke({"question": user_message})
    history = get_session_history(session_id)
    history_aware_retriever = get_history_retriever()
    docs = history_aware_retriever.invoke(
        {"input": normalized_question, "chat_history": history.messages}
    )
    return [
        {
            "text": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in docs
    ]


def get_ai_response(user_message, session_id: str = "abc123"):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    retrieved_context = get_retrieved_context(user_message, session_id)
    raw_chain = {"input": dictionary_chain} | rag_chain

    ai_response = raw_chain.stream(
        {"question": user_message},
        config={
            "configurable": {"session_id": session_id}
        },
    )

    return ai_response, retrieved_context
