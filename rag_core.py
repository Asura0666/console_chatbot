import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ======================================================================
# FILE PURPOSE: RAG Pipeline Setup and Document Processing
#
# This file contains the logic for:
# 1. Loading, splitting, and embedding documents into a FAISS vector store.
# 2. Constructing the core LangChain Expression Language (LCEL) chain.
# 3. Integrating the dynamic retriever and conversational memory.
# ======================================================================


def load_and_process_document(file_path: str):
    """
    Loads a local document from the specified path, splits it into chunks,
    creates embeddings using the Gemini model, and stores them in a FAISS
    vector index.

    The final output is a LangChain Retriever object, ready to be placed
    into the RetrieverHolder for RAG-enabled chat.

    Supported formats: .txt, .pdf, .docx, and .csv.

    Args:
        file_path (str): The local path to the document file.

    Returns:
        FAISS.as_retriever() or None: A retriever instance if successful, or None if
                                      loading/processing fails due to errors or unsupported format.
    """
    print(f"Loading document from: {file_path}...")

    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            print(
                "Error: Unsupported file format. Please use .txt, .pdf, .docx, or .csv"
            )
            return None

        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, 
        )
        splits = text_splitter.split_documents(docs)
        print(f"Document split into {len(splits)} chunks.")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        return vectorstore.as_retriever()

    except ImportError as e:
        print(f"\n--- Dependency Missing Error ---")
        if (
            "PyPDFLoader" in str(e)
            or file_path.endswith(".pdf")
            or file_path.endswith(".csv")
        ):
            print("To process PDF or CSV files, you likely need to install 'pypdf'.")
            print("Please run: pip install pypdf")
        elif "UnstructuredWordDocumentLoader" in str(e) or file_path.endswith(".docx"):
            print(
                "To process DOCX files, you need to install 'unstructured' and its dependencies."
            )
            print("Please run: pip install unstructured[docx]")
        else:
            print(
                f"An ImportError occurred: {e}. Check if you have all required document parser libraries installed."
            )
        return None

    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        if "Quota exceeded" in str(e):
            print("\n--- RAG MODE FAILED DUE TO QUOTA ---")
            print(
                "NOTE: The document failed to load because the Gemini API quota for embeddings was exceeded."
            )
            print(
                "This often happens when processing many chunks (like a large CSV/PDF)."
            )
            print("Please wait for your quota to reset or check your API usage/plan.")
        return None


def setup_rag_chain(retriever_holder, get_session_history_func):
    """
    Sets up the LangChain pipeline (LCEL), including the model, prompt, and
    conversational memory wrapper.

    Args:
        retriever_holder (RetrieverHolder): The mutable object holding the active retriever.
        get_session_history_func (function): The function (from utils.py) to manage chat history persistence.

    Returns:
        RunnableWithMessageHistory: The complete, runnable conversational RAG chain.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.3, convert_system_message_to_human=True
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using the provided context if it is relevant. "
                "If the context is not sufficient or the question is general, use your general knowledge to answer.\n\nContext:\n{context}",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        """Converts a list of LangChain Document objects into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def dynamic_retrieval(x):
        """
        Retrieves context using the retriever currently stored in the holder.
        This is the core of the dynamic switching mechanism.
        """
        return format_docs(retriever_holder.retriever.invoke(x["question"]))

    contextual_chain = RunnablePassthrough.assign(context=dynamic_retrieval)

    full_rag_chain = contextual_chain | prompt | llm | StrOutputParser()

    conversational_rag_chain = RunnableWithMessageHistory(
        full_rag_chain,
        get_session_history_func,
        input_messages_key="question",
        history_messages_key="history",
    )

    return conversational_rag_chain
