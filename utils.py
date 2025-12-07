import os
import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# ======================================================================
# FILE PURPOSE: Session Management and Utility Classes
#
# This file handles two main responsibilities:
# 1. Persistence: Saving and loading chat history to a local JSON file.
# 2. Dynamic RAG State: Providing classes to manage the active document retriever.
# ======================================================================


SESSION_HISTORY_DIR = "chat_history"
os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)

store = {}


class RetrieverHolder:
    """
    A mutable container class designed to hold the currently active LangChain
    retriever instance.

    This is critical because the RAG chain is defined once at startup, but the
    retriever (the source of truth) needs to be dynamically swapped at runtime
    when a user loads a new document via the `/load` command.
    """

    def __init__(self, retriever):
        """
        Initializes the holder with an initial retriever instance.
        Typically initialized with DummyRetriever for general chat mode.
        """
        self.retriever = retriever


class DummyRetriever:
    """
    Implements the retriever interface but returns an empty list of documents.

    When the RetrieverHolder contains this instance, the RAG part of the
    LangChain expression is effectively disabled, forcing the model to rely
    only on its general knowledge and chat history (pure LLM mode).
    """

    def invoke(self, query):
        """
        Mimics the LangChain retriever 'invoke' method but returns no context.
        """
        return []


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieves the chat history for a specific session ID, ensuring persistence.

    If the session ID is new or not in the in-memory store, it attempts to load
    the history from the corresponding JSON file.

    Args:
        session_id (str): A unique identifier for the user's conversation session
                          (e.g., 'user_session_v1' or an actual authenticated user ID).

    Returns:
        InMemoryChatMessageHistory: The loaded or newly initialized history object.

    Notes on Storage:
    1. Only one session file is created per 'session_id'. Since we use a fixed
       'user_session_v1', only one file will exist for all runs. For multiple
       users, this ID would need to be dynamic.
    2. We store the *exact* conversation. For extremely long conversations, this
       could become too large, and an advanced memory technique like summarization
       or context compression would be required for efficiency.
    """
    if session_id not in store:
        history_file = os.path.join(SESSION_HISTORY_DIR, f"{session_id}.json")
        history = InMemoryChatMessageHistory()

        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    for message_data in data:
                        if message_data["type"] == "human":
                            history.add_message(
                                HumanMessage(content=message_data["content"])
                            )
                        elif message_data["type"] == "ai":
                            history.add_message(
                                AIMessage(content=message_data["content"])
                            )

            except Exception as e:
                pass

        store[session_id] = history
    return store[session_id]


def save_session_history(session_id: str):
    """
    Saves the current chat history for a session to a JSON file for persistence.
    This function should be called after every successful conversational turn and
    upon application exit.

    Args:
        session_id (str): The identifier for the session whose history needs saving.
    """
    if session_id in store:
        history = store[session_id]
        history_file = os.path.join(SESSION_HISTORY_DIR, f"{session_id}.json")

        serializable_history = [
            {"type": msg.type, "content": msg.content} for msg in history.messages
        ]

        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, indent=4)
        except Exception as e:
            print(f"Error saving history to file: {e}")
