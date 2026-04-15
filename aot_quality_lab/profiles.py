from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from runtime import PROJECT_ROOT


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    device: str = "cpu"
    max_length: int = 512
    trust_remote_code: bool = False


@dataclass(frozen=True)
class VectorstoreConfig:
    collection_name: str
    persist_directory: str
    source_glob: str | None = None


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    description: str
    chat: ChatConfig
    embedding: EmbeddingConfig
    vectorstore: VectorstoreConfig
    retriever_k: int = 10
    multi_query: bool = False
    query_variant_count: int = 3

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _openrouter_headers() -> dict[str, str]:
    headers = {}
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
    x_title = os.getenv("OPENROUTER_X_TITLE")
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title
    return headers


def _default_profiles() -> dict[str, PipelineProfile]:
    return {
        "upstage_prod": PipelineProfile(
            name="upstage_prod",
            description="기존 서비스와 동일한 Upstage 생성/임베딩 + AoT 인덱스",
            chat=ChatConfig(provider="upstage", model="solar-pro2"),
            embedding=EmbeddingConfig(
                provider="upstage",
                model="solar-embedding-1-large",
            ),
            vectorstore=VectorstoreConfig(
                collection_name="AoT",
                persist_directory=str(PROJECT_ROOT / "AoT"),
                source_glob="data/attack_on_Titan_Namu_new_part*.jsonl",
            ),
        ),
        "openrouter_gemma_free_existing_aot": PipelineProfile(
            name="openrouter_gemma_free_existing_aot",
            description="OpenRouter 무료 Gemma 생성 모델 + 기존 Upstage AoT 인덱스",
            chat=ChatConfig(
                provider="openai_compatible",
                model=os.getenv(
                    "AOT_OPENROUTER_CHAT_MODEL",
                    "google/gemma-4-31b-it:free",
                ),
                api_key_env="OPENROUTER_API_KEY",
                base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                default_headers=_openrouter_headers(),
            ),
            embedding=EmbeddingConfig(
                provider="upstage",
                model="solar-embedding-1-large",
            ),
            vectorstore=VectorstoreConfig(
                collection_name="AoT",
                persist_directory=str(PROJECT_ROOT / "AoT"),
                source_glob="data/attack_on_Titan_Namu_new_part*.jsonl",
            ),
        ),
        "openrouter_gemma_free_local_e5": PipelineProfile(
            name="openrouter_gemma_free_local_e5",
            description="OpenRouter 무료 Gemma 생성 모델 + 로컬 multilingual-e5 임베딩 인덱스",
            chat=ChatConfig(
                provider="openai_compatible",
                model=os.getenv(
                    "AOT_OPENROUTER_CHAT_MODEL",
                    "google/gemma-4-31b-it:free",
                ),
                api_key_env="OPENROUTER_API_KEY",
                base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                default_headers=_openrouter_headers(),
            ),
            embedding=EmbeddingConfig(
                provider="local_hf",
                model=os.getenv(
                    "AOT_LOCAL_EMBED_MODEL",
                    "intfloat/multilingual-e5-base",
                ),
                query_prefix="query: ",
                document_prefix="passage: ",
                normalize=True,
                device=os.getenv("AOT_LOCAL_EMBED_DEVICE", "cpu"),
                max_length=int(os.getenv("AOT_LOCAL_EMBED_MAX_LENGTH", "512")),
            ),
            vectorstore=VectorstoreConfig(
                collection_name="AoT_e5",
                persist_directory=str(PROJECT_ROOT / "AoT_e5"),
                source_glob="data/attack_on_Titan_Namu_new_part*.jsonl",
            ),
        ),
    }


def get_default_profile_name() -> str:
    if os.getenv("AOT_PROFILE"):
        return os.environ["AOT_PROFILE"]
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter_gemma_free_existing_aot"
    return "upstage_prod"


def list_profiles() -> list[PipelineProfile]:
    return [resolve_profile(name) for name in get_profile_names()]


def get_profile_names() -> list[str]:
    return sorted(_default_profiles().keys())


def _apply_env_overrides(profile: PipelineProfile) -> PipelineProfile:
    chat = profile.chat
    embedding = profile.embedding
    vectorstore = profile.vectorstore

    if os.getenv("AOT_CHAT_PROVIDER"):
        chat = replace(chat, provider=os.environ["AOT_CHAT_PROVIDER"])
    if os.getenv("AOT_CHAT_MODEL"):
        chat = replace(chat, model=os.environ["AOT_CHAT_MODEL"])
    if os.getenv("AOT_CHAT_API_KEY_ENV"):
        chat = replace(chat, api_key_env=os.environ["AOT_CHAT_API_KEY_ENV"])
    if os.getenv("AOT_CHAT_BASE_URL"):
        chat = replace(chat, base_url=os.environ["AOT_CHAT_BASE_URL"])
    if os.getenv("AOT_CHAT_TEMPERATURE"):
        chat = replace(chat, temperature=float(os.environ["AOT_CHAT_TEMPERATURE"]))
    if os.getenv("AOT_CHAT_MAX_TOKENS"):
        chat = replace(chat, max_tokens=int(os.environ["AOT_CHAT_MAX_TOKENS"]))

    if os.getenv("AOT_EMBED_PROVIDER"):
        embedding = replace(embedding, provider=os.environ["AOT_EMBED_PROVIDER"])
    if os.getenv("AOT_EMBED_MODEL"):
        embedding = replace(embedding, model=os.environ["AOT_EMBED_MODEL"])
    if os.getenv("AOT_EMBED_API_KEY_ENV"):
        embedding = replace(
            embedding,
            api_key_env=os.environ["AOT_EMBED_API_KEY_ENV"],
        )
    if os.getenv("AOT_EMBED_BASE_URL"):
        embedding = replace(embedding, base_url=os.environ["AOT_EMBED_BASE_URL"])
    if os.getenv("AOT_EMBED_QUERY_PREFIX"):
        embedding = replace(
            embedding,
            query_prefix=os.environ["AOT_EMBED_QUERY_PREFIX"],
        )
    if os.getenv("AOT_EMBED_DOCUMENT_PREFIX"):
        embedding = replace(
            embedding,
            document_prefix=os.environ["AOT_EMBED_DOCUMENT_PREFIX"],
        )
    if os.getenv("AOT_EMBED_DEVICE"):
        embedding = replace(embedding, device=os.environ["AOT_EMBED_DEVICE"])
    if os.getenv("AOT_EMBED_MAX_LENGTH"):
        embedding = replace(
            embedding,
            max_length=int(os.environ["AOT_EMBED_MAX_LENGTH"]),
        )
    if os.getenv("AOT_EMBED_NORMALIZE"):
        embedding = replace(
            embedding,
            normalize=_env_flag("AOT_EMBED_NORMALIZE", embedding.normalize),
        )

    if os.getenv("AOT_COLLECTION_NAME"):
        vectorstore = replace(
            vectorstore,
            collection_name=os.environ["AOT_COLLECTION_NAME"],
        )
    if os.getenv("AOT_PERSIST_DIRECTORY"):
        vectorstore = replace(
            vectorstore,
            persist_directory=_resolve_path(os.environ["AOT_PERSIST_DIRECTORY"]),
        )
    if os.getenv("AOT_SOURCE_GLOB"):
        vectorstore = replace(vectorstore, source_glob=os.environ["AOT_SOURCE_GLOB"])

    retriever_k = int(os.getenv("AOT_RETRIEVER_K", str(profile.retriever_k)))
    multi_query = _env_flag("AOT_MULTI_QUERY", profile.multi_query)
    query_variant_count = int(
        os.getenv("AOT_QUERY_VARIANT_COUNT", str(profile.query_variant_count))
    )

    return PipelineProfile(
        name=profile.name,
        description=profile.description,
        chat=chat,
        embedding=embedding,
        vectorstore=vectorstore,
        retriever_k=retriever_k,
        multi_query=multi_query,
        query_variant_count=query_variant_count,
    )


@lru_cache(maxsize=16)
def resolve_profile(name: str | None = None) -> PipelineProfile:
    profiles = _default_profiles()
    profile_name = name or get_default_profile_name()
    if profile_name not in profiles:
        raise KeyError(
            f"Unknown profile '{profile_name}'. Available profiles: {', '.join(sorted(profiles))}"
        )
    return _apply_env_overrides(profiles[profile_name])


def _require_env(name: str | None) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is required.")
    return value


def build_chat_model(profile_name: str | None = None):
    profile = resolve_profile(profile_name)
    chat = profile.chat

    if chat.provider == "upstage":
        from langchain_upstage import ChatUpstage

        kwargs = {"model": chat.model, "temperature": chat.temperature}
        if chat.max_tokens is not None:
            kwargs["max_tokens"] = chat.max_tokens
        return ChatUpstage(**kwargs)

    if chat.provider in {"openai", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": chat.model,
            "temperature": chat.temperature,
            "api_key": _require_env(chat.api_key_env),
            "base_url": chat.base_url,
        }
        if chat.default_headers:
            kwargs["default_headers"] = chat.default_headers
        if chat.max_tokens is not None:
            kwargs["max_tokens"] = chat.max_tokens
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported chat provider: {chat.provider}")


def build_embedding_model(profile_name: str | None = None):
    profile = resolve_profile(profile_name)
    embedding = profile.embedding

    if embedding.provider == "upstage":
        from langchain_upstage import UpstageEmbeddings

        return UpstageEmbeddings(model=embedding.model)

    if embedding.provider in {"openai", "openai_compatible"}:
        from langchain_openai import OpenAIEmbeddings

        kwargs = {
            "model": embedding.model,
            "api_key": _require_env(embedding.api_key_env),
            "base_url": embedding.base_url,
        }
        return OpenAIEmbeddings(**kwargs)

    if embedding.provider == "local_hf":
        from local_embeddings import LocalHFEmbeddings

        return LocalHFEmbeddings(
            model_name=embedding.model,
            device=embedding.device,
            normalize=embedding.normalize,
            query_prefix=embedding.query_prefix,
            document_prefix=embedding.document_prefix,
            max_length=embedding.max_length,
            trust_remote_code=embedding.trust_remote_code,
        )

    raise ValueError(f"Unsupported embedding provider: {embedding.provider}")


def vectorstore_exists(profile_name: str | None = None) -> bool:
    profile = resolve_profile(profile_name)
    persist_directory = Path(profile.vectorstore.persist_directory)
    return persist_directory.exists() and any(persist_directory.iterdir())


def ensure_vectorstore_ready(profile_name: str | None = None) -> None:
    profile = resolve_profile(profile_name)
    if vectorstore_exists(profile.name):
        return

    message = [
        f"Vectorstore for profile '{profile.name}' was not found.",
        f"Expected directory: {profile.vectorstore.persist_directory}",
    ]
    if profile.vectorstore.source_glob:
        message.append(
            "Build it with: "
            f"/mnt/e/one_piece/venv/bin/python /mnt/e/one_piece/aot_quality_lab/build_vectorstore.py --profile {profile.name}"
        )
    raise RuntimeError(" ".join(message))
