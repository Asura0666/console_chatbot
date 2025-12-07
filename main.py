import os
from dotenv import load_dotenv

from utils import (
    RetrieverHolder,      
    DummyRetriever,       
    get_session_history,  
    save_session_history,
)
from rag_core import load_and_process_document, setup_rag_chain

load_dotenv()


def main():
    """
    Main execution loop for the RAG Chatbot.

    This function handles the application lifecycle:
    1. Configuration check (API Key).
    2. Initialization of the RAG chain and dynamic retriever.
    3. Management of the conversational session ID.
    4. Continuous loop for user interaction, including command handling (/load)
       and chat processing (chain invocation).
    5. Persistence of chat history on every turn and on exit.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        print("FATAL ERROR: Please set your GOOGLE_API_KEY environment variable.")
        return

    print("=" * 60)
    print("--- Gemini 2.5 Flash RAG Chatbot ---")
    print(
        "Welcome! This chatbot can answer general questions or questions based on your documents."
    )
    print("=" * 60)
    print("To get started, here are the commands you can use:")
    print(" - Type your question for general chat.")
    print(" - Type `/load <path/to/your/file.pdf>` to activate RAG mode.")
    print(" - Type `quit` or `exit` to end the session.")
    print("-" * 60)

    retriever_holder = RetrieverHolder(DummyRetriever())

    conversational_rag_chain = setup_rag_chain(retriever_holder, get_session_history)

    session_id = "user_session_id_v1"

    print("Chatbot is ready. Starting conversation.")

    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            user_input = "quit"

        if user_input.lower() in ["quit", "exit"]:
            save_session_history(session_id)
            print("Exiting...")
            break

        if user_input.lower().startswith("/load "):
            new_doc_path = user_input.split(" ", 1)[-1].strip()

            new_doc_path = os.path.normpath(new_doc_path)

            if not os.path.exists(new_doc_path):
                print(
                    f"File not found at: {new_doc_path}. Current RAG status remains unchanged."
                )
                continue

            new_retriever = load_and_process_document(new_doc_path)

            if new_retriever:
                retriever_holder.retriever = new_retriever
                print(
                    f"\n--- RAG MODE ACTIVATED ---\nSuccessfully loaded {new_doc_path}. You can now ask questions about this document."
                )
            else:
                print(
                    f"\n--- RAG MODE FAILED ---\nFailed to load document at {new_doc_path}. Reverting to general chat mode."
                )
                retriever_holder.retriever = DummyRetriever()

            print("-" * 60)
            continue

        try:
            response = conversational_rag_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Gemini: {response}")

            save_session_history(session_id)

            print("-" * 60)

        except Exception as e:
            print(f"An error occurred during chat: {e}")
            print(
                "Please try your question again or check your API key/network connection."
            )


if __name__ == "__main__":
    main()

