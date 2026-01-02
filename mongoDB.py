
import os
from datetime import datetime, timezone
from typing import Optional, Sequence

from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Add it to your environment or .env file.")

# Create a new client and connect to the server
client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))

CHAT_LOG_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "session_id",
            "user_query",
            "ai_response",
            "retrieved_context",
            "timestamp",
        ],
        "properties": {
            "session_id": {
                "bsonType": "string",
                "description": "User session identifier",
            },
            "user_query": {"bsonType": "string"},
            "ai_response": {"bsonType": "string"},
            "retrieved_context": {
                "bsonType": "array",
                "items": {"bsonType": "string"},
                "description": "Retrieved document snippets",
            },
            "feedback": {
                "bsonType": "string",
                "description": "Optional like/dislike feedback",
            },
            "timestamp": {"bsonType": "date"},
        },
    }
}


def get_collection(
    db_name: str = "attackTitan",
    collection_name: str = "chat_logs",
) -> Collection:
    db = client[db_name]
    if collection_name not in db.list_collection_names():
        db.create_collection(
            collection_name,
            validator=CHAT_LOG_SCHEMA,
            validationLevel="moderate",
        )
    else:
        db.command(
            "collMod",
            collection_name,
            validator=CHAT_LOG_SCHEMA,
            validationLevel="moderate",
        )
    return db[collection_name]


def insert_chat_log(
    session_id: str,
    user_query: str,
    ai_response: str,
    retrieved_context: Optional[Sequence[str]] = None,
    feedback: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    db_name: str = "attackTitan",
    collection_name: str = "chat_logs",
):
    collection = get_collection(db_name=db_name, collection_name=collection_name)
    doc = {
        "session_id": session_id,
        "user_query": user_query,
        "ai_response": ai_response,
        "retrieved_context": list(retrieved_context or []),
        "timestamp": timestamp or datetime.now(timezone.utc),
    }
    if feedback is not None:
        doc["feedback"] = feedback
    return collection.insert_one(doc)


def ping() -> None:
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as exc:
        print(exc)


if __name__ == "__main__":
    ping()
