from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from dotenv import load_dotenv

from config import answer_examples

load_dotenv()



store = {}

REPLACEMENTS = {
    "사람을 나타내는 표현": "진격의거인 애니메이션 안의 등장인물이나 사람들",
}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print(store)
    return store[session_id]


@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(model='text-embedding-3-small')


@lru_cache(maxsize=1)
def get_retriever():
    # database = Chroma(collection_name='chroma-inu-new', persist_directory="./chroma_inu-new", embedding_function=get_embeddings())
    database = Chroma(collection_name='attackTitan', persist_directory="./attackTitan", embedding_function=get_embeddings())
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


def get_llm(model='gpt-4o-mini'):
    llm = ChatOpenAI(model=model)
    # llm = Ollama(model=model)
    return llm


def normalize_question(question: str) -> str:
    normalized = question
    for source, target in REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


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
    system_prompt = (
    """당신은 애니메이션 및 원작 만화 『진격의 거인』(Attack on Titan)의 설정, 스토리, 인물, 세계관에 대해
    정확하고 깊이 있는 지식을 가진 전문가입니다.
    사용자의 질문에 대해 오직 『진격의 거인』 작품 내부 정보에 기반하여서만 답변해야 합니다.

    [중요 규칙 1]
    사용자의 질문이 『진격의 거인』과 직접적으로 관련되지 않은 경우,
    추측하거나 일반 상식으로 답변하지 말고 반드시 다음과 같이 답변하십시오:
    "모르겠습니다. 진격의 거인과 관련된 질문만 답변할 수 있습니다."

    [중요 규칙 2 - 이름 및 용어 표기 통일]
    사용자가 번역 차이, 음역 차이, 오타, 별칭 등으로 질문하더라도
    아래의 대응 관계를 자동으로 동일 개체로 인식해야 합니다.
    단, 답변에서는 반드시 한국 공식 번역 기준의 정식 명칭만 사용하십시오.

    ── 인물 ──
    - 에렌 예거: 에렌, 엘런, 엘렌, 엘런 예거, 엘렌 예거, Eren, Yeager, Jaeger
    - 미카사 아커만: 미카사, 미카사 아카만, 아커만 미카사, Mikasa
    - 리바이 아커만: 리바이, 리바이 병장, Levi
    - 아르민 알레르토: 아르민, 알민, Armin
    - 라이너 브라운: 라이너, Reiner
    - 베르톨트 후버: 베르톨트, 베르톨트 후버, 베르톨트 훼버, Bertolt, Bertholdt
    - 애니 레온하트: 애니, Annie
    - 히스토리아 레이스: 히스토리아, 크리스타, Historia, Krista
    - 지크 예거: 지크, 지크 예거, Zeke
    - 한지 조에: 한지, 한지 조에, Hange
    - 에르빈 스미스: 에르빈, Erwin
    - 피크 핑거: 피크, Pieck
    - 로드 레이스: 론고, 론고 레이스, 로드
    
    ── 거인 ──
    - 차력 거인: 수레 거인, 카트 거인, Cart Titan
    - 시조의 거인: Founding Titan
    - 진격의 거인: 공격 거인, Attack Titan
    - 초대형 거인: Colossal Titan
    - 갑옷 거인: Armored Titan
    - 여성형 거인: Female Titan
    - 짐승 거인: Beast Titan
    - 턱 거인: Jaw Titan
    - 전퇴의 거인: War Hammer Titan

    ── 가문 / 조직 ──
    - 타이버 가문: 티부르 가문, 티부어 가문, Tybur family, Tybur, 티바 가문
    - 아커만 가문: 아커맨 가문, 액커맨 가문, 아커만 가문

    [중요 규칙 3]
    인물, 사건, 설정에 대해 답변할 때는
    - 작중 시점
    - 애니메이션 / 만화 기준 여부
    - 명확히 밝혀진 설정과 해석의 영역을 구분
    하여 서술하십시오.

    [중요 규칙 4]
    확실하지 않거나 작중에서 명시되지 않은 내용은
    사실처럼 단정하지 말고 "작중에서 명확히 밝혀지지 않았다"고 답변하십시오.

    [중요 규칙 5]
    - 사용자가 특정 등장인물의 말투(예: 아르민, 리바이 등)로 답변해달라고 요청할 경우,
        해당 캐릭터의 성격, 사고방식, 말투를 반영하여 답변하세요.
    - 말투 요청이 없는 경우에는 중립적이고 설명적인 어투를 유지하세요.

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
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain


def get_retrieved_context(user_message: str, session_id: str):
    normalized_question = normalize_question(user_message)
    docs = retrieve_docs(normalized_question, session_id)
    return [
        {
            "text": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in docs
    ]


def retrieve_docs(question: str, session_id: str):
    history = get_session_history(session_id)
    if len(history.messages) >= 2:
        history_aware_retriever = get_history_retriever()
        return history_aware_retriever.invoke(
            {"input": question, "chat_history": history.messages}
        )
    return get_retriever().invoke(question)


def _stream_answer(chain, inputs: dict, history: BaseChatMessageHistory, user_message: str):
    buffer = []
    for chunk in chain.stream(inputs):
        buffer.append(chunk)
        yield chunk
    answer = "".join(buffer)
    history.add_user_message(user_message)
    history.add_ai_message(answer)


def get_ai_response(user_message, session_id: str = "abc123"):
    normalized_question = normalize_question(user_message)
    docs = retrieve_docs(normalized_question, session_id)
    retrieved_context = [
        {
            "text": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in docs
    ]

    history = get_session_history(session_id)
    answer_chain = get_answer_chain()
    inputs = {
        "input": normalized_question,
        "context": docs,
        "chat_history": history.messages,
    }
    ai_response = _stream_answer(answer_chain, inputs, history, user_message)

    return ai_response, retrieved_context
