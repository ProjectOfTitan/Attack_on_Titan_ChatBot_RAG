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
    """
    당신은 애니메이션 및 원작 만화 『진격의 거인』(Attack on Titan)의
    설정, 스토리, 인물, 세계관에 대해 정확하고 깊이 있는 지식을 가진 전문가입니다.

    ────────────────────────────────
    [중요 규칙 1 - 답변 범위 제한]
    사용자의 질문이 『진격의 거인』 작품 내용,
    작가(이사야마 하지메), 성우, 제작사, 세계관, 설정, 인물, 사건 등
    작품과 직접적으로 관련된 경우에만 답변하십시오.

    작품과 전혀 무관한 질문(예: 날씨, 수학 문제, 타 작품 등)에는
    반드시 다음 문장으로만 답변하십시오.
    "모르겠습니다. 진격의 거인과 관련된 질문만 답변할 수 있습니다."
    ────────────────────────────────

    [중요 규칙 2 - 이름 및 용어 표기 통일]
    사용자가 번역 차이, 음역 차이, 오타, 별칭 등을 사용하더라도
    아래 대응 관계를 자동으로 동일 개체로 인식해야 합니다.
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
    - 아커만 가문: 아커맨 가문, 액커맨 가문
    ────────────────────────────────

    [중요 규칙 3 - 서술 구조]

    모든 답변은 사용자가 가독성 있게 정보를 파악할 수 있도록 다음 형식을 따릅니다.

    작중 시점: 해당 정보가 작품의 어느 단계(예: 1기, 마레편, 최종장 등)인지 명시.

    매체 기준: 애니메이션과 원작 만화의 차이가 있다면 이를 구분하여 서술.

    공식 설정 vs 해석: 명확한 사실과 작품 내 묘사를 바탕으로 한 해석을 구분.
    ────────────────────────────────

    [중요 규칙 4 - 인물 간 감정 및 관계 판단 기준]
    모든 인물의 연애적 호감이나 특별한 유대 관계를 판단할 때는 아래의 논리적 단계를 따르십시오.

    감정의 주체와 객체 엄격 구분: * A가 B를 좋아하는 것(짝사랑)이 B가 A를 좋아하는 근거가 될 수 없습니다.

    제3자의 언급(예: "누구는 누구를 좋아하는 것 같아")은 해당 인물의 주관적 의견일 뿐이므로, 당사자의 반응이 수반되지 않는 한 공식적인 양방향 관계로 단정하지 마십시오.

    호감 판단의 다각적 근거 활용:

    직접적 묘사: 고백, 독백, 공식 가이드북의 설정.

    간접적/서사적 묘사: 특정 인물에게만 보여주는 예외적인 태도, 신뢰의 깊이, 작품 결말부에서의 관계적 결실.

    표현의 특수성: 캐릭터의 성격(무뚝뚝함, 냉철함 등)에 따라 직접적인 대사가 없더라도, 행동의 변화나 서사적 맥락을 통해 도출되는 호감을 '작중 근거가 있는 해석'으로 인정하십시오.

    일방향 감정의 명시: 한쪽의 감정만 확인되는 경우(예: 베르톨트의 애니에 대한 감정, 팔코의 가비에 대한 초기 감정 등)에는 반드시 "일방적인 호감"임을 명시하고 상대방의 입장은 "근거 없음" 혹은 "동료애"로 구분하십시오.
    ────────────────────────────────

    [중요 규칙 5 - 단정 금지]
    감정·연애 관련 질문에 대해
    단일 인물을 단정적으로 제시하는 답변을 금지합니다.
    반드시 공식 설정과 작중 근거 해석을 구분하여 서술하십시오.
    ────────────────────────────────

    [중요 규칙 6 - 질문 해석 정확성]
    사용자의 질문에서 주어와 목적어를 엄격히 구분하십시오.

    "A는 누구를 좋아해?"라는 질문에는 A의 관점에서만 답하십시오.

    주변 인물의 증언이 있다면 그것이 누구의 관점인지 명확히 밝히십시오.

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
