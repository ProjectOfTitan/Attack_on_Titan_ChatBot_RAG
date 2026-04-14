from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from profiles import build_embedding_model, resolve_profile
from runtime import PROJECT_ROOT


def _stable_id(record: dict, source_file: Path) -> str:
    raw = "|".join(
        [
            str(source_file),
            str(record.get("title", "")),
            str(record.get("section", "")),
            str(record.get("chunk_index", "")),
            str(record.get("text", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_documents(glob_pattern: str, limit: int | None = None) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    files = sorted(PROJECT_ROOT.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = str(record.get("text", "")).strip()
                if not text:
                    continue
                metadata = {key: value for key, value in record.items() if key != "text"}
                metadata["source_file"] = str(file_path)
                documents.append(Document(page_content=text, metadata=metadata))
                ids.append(_stable_id(record, file_path))
                if limit is not None and len(documents) >= limit:
                    return documents, ids

    return documents, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--glob")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    profile = resolve_profile(args.profile)
    glob_pattern = args.glob or profile.vectorstore.source_glob
    if not glob_pattern:
        raise ValueError("No source glob was provided for building the vectorstore.")

    persist_directory = Path(profile.vectorstore.persist_directory)
    if args.recreate and persist_directory.exists():
        shutil.rmtree(persist_directory)

    documents, ids = load_documents(glob_pattern, limit=args.limit)
    embedding_model = build_embedding_model(profile.name)
    vectorstore = Chroma(
        collection_name=profile.vectorstore.collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding_model,
    )

    for start in range(0, len(documents), args.batch_size):
        end = start + args.batch_size
        vectorstore.add_documents(documents=documents[start:end], ids=ids[start:end])

    print(
        json.dumps(
            {
                "profile": profile.name,
                "documents": len(documents),
                "collection_name": profile.vectorstore.collection_name,
                "persist_directory": str(persist_directory),
                "source_glob": glob_pattern,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
