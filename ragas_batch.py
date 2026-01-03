from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

load_dotenv()

EVAL_MODEL = "gpt-5.1"
EVAL_EMBED_MODEL = "text-embedding-3-small"
VERBOSE = True
LOG_PATH = "ragas_answers.json"
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 12
RETRIEVER_LAMBDA = 0.7

QUESTIONS = [
    "마레와 파라디섬은 무슨 사건때문에 갈등이 생겨났어?",
    "에렌이 무슨 목적으로 땅울림을 시전했어?",
    "아홉거인에 대한 간단한 설명과 특징을 말해줘",
    "짐승 거인의 능력은 계승자에 따라서 어떻게 달라져?",
    "라이너가 본인이 갑옷거인이라고 처음 말한건 애니메이션 몇 화에 나와?",
    "시간시나 구의 붕괴 당시 초대형 거인을 조종한 캐릭터는 누구야?",
    "만화에 나오는 조사병단, 헌병단, 주둔병단은 각각 어떤 역할을 맡고 있어?",
    "미카사가 에렌을 유독 챙기는 이유가 뭐야?",
    "구 리바이반의 멤버들의 이름을 알려줘",
    "레이스가문이 숨기고 있는 것의 정체가 뭐야?",
]

GROUND_TRUTHS = [
    
]

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    database = Chroma(
        collection_name="attackTitan",
        persist_directory="./attackTitan",
        embedding_function=embedding,
    )
    retriever = database.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA,
        },
    )
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


def get_llm(model="gpt-5-mini"):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 진격의거인 애니메이션 안의 등장인물이나 사람들"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """
    )

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


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
        """당신은 RAG 어시스턴트입니다. 아래 컨텍스트만 사용해 답하세요.
규칙:
1) 컨텍스트에 명시된 사실만 사용한다. 추정/상식/외부지식 금지.
2) 컨텍스트 내용을 재서술/요약해서 답할 수 있으나, 새로운 사실은 추가하지 않는다.
3) 답변 각 주장에 대해 컨텍스트 근거를 함께 제시한다.
4) 컨텍스트가 부족하면 "정보 부족"이라고 말하고 필요한 추가 질문을 1개 제안한다.
5) 불확실한 표현(아마, 추측)은 사용하지 않는다.
6) 한국어로 간결하게 답한다.

컨텍스트:
{context}

질문:
{input}

출력 형식:
- 답변: ...
- 근거: [컨텍스트 인용]
- 부족한 점: (없으면 "없음")"""
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
    ).pick("answer")

    return conversational_rag_chain


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


def _normalize_contexts(retrieved_context: list[dict]) -> list[dict]:
    seen = set()
    normalized = []
    for item in retrieved_context:
        text = item.get("text", "").strip()
        if len(text) < 30:
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(item)
    return normalized


def get_retrieved_context(user_message: str, session_id: str):
    dictionary_chain = get_dictionary_chain()
    normalized_question = dictionary_chain.invoke({"question": user_message})
    history = get_session_history(session_id)
    history_aware_retriever = get_history_retriever()
    docs = history_aware_retriever.invoke(
        {"input": normalized_question, "chat_history": history.messages}
    )
    retrieved = [
        {
            "text": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in docs
    ]
    return _normalize_contexts(retrieved)


def get_ai_response(user_message: str, session_id: str = "ragas_batch"):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    raw_chain = {"input": dictionary_chain} | rag_chain
    ai_response = raw_chain.invoke(
        {"question": user_message},
        config={"configurable": {"session_id": session_id}},
    )
    return ai_response


def generate_batch_answers(questions: list[str], limit: int = 10):
    results = []
    for idx, question in enumerate(questions[:limit]):
        session_id = f"ragas_batch_{idx}"
        if VERBOSE:
            print(f"[{idx + 1}/{min(len(questions), limit)}] Generating answer...")
        retrieved_context = get_retrieved_context(question, session_id)
        answer = get_ai_response(question, session_id)
        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [item["text"] for item in retrieved_context],
                "retrieved_context": retrieved_context,
            }
        )
    return results


def build_dataset(results: list[dict]):
    dataset_dict = {
        "question": [item["question"] for item in results],
        "answer": [item["answer"] for item in results],
        "contexts": [item["contexts"] for item in results],
    }
    if GROUND_TRUTHS and any(text.strip() for text in GROUND_TRUTHS):
        trimmed = GROUND_TRUTHS[: len(results)]
        dataset_dict["ground_truth"] = trimmed
        dataset_dict["reference"] = trimmed

    try:
        from datasets import Dataset
    except Exception:
        return dataset_dict, None

    dataset = Dataset.from_dict(dataset_dict)
    return dataset_dict, dataset


def evaluate_with_ragas(dataset):
    try:
        from ragas import evaluate
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness
    except Exception as exc:
        print(f"ragas import failed: {exc}")
        return None

    evaluator_llm = ChatOpenAI(model=EVAL_MODEL)
    evaluator_embeddings = OpenAIEmbeddings(model=EVAL_EMBED_MODEL)
    metrics = [faithfulness, answer_relevancy]
    if "ground_truth" in dataset.features or "reference" in dataset.features:
        metrics.append(context_precision)
        metrics.append(context_recall)

    if VERBOSE:
        print("Running RAGAS evaluation...")
    return evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )


def main() -> None:
    results = generate_batch_answers(QUESTIONS, limit=10)
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] {item['question']}")
        print(item["answer"])
        print("-" * 80)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        import json

        json.dump(results, log_file, ensure_ascii=False, indent=2)

    dataset_dict, dataset = build_dataset(results)
    if dataset is None:
        print("datasets 미설치: Dataset 대신 dict를 출력합니다.")
        print(dataset_dict)
        return

    print(dataset)
    ragas_result = evaluate_with_ragas(dataset)
    if ragas_result is not None:
        print(ragas_result)


if __name__ == "__main__":
    main()
