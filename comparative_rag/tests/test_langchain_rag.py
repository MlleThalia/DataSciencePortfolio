import pytest
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from comparative_rag.scripts.langchain_rag import LangChainRAG
from comparative_rag.ui.utils.utils import chat_mistral_ai

mistral_client = chat_mistral_ai()
LANGCHAIN_RAG = LangChainRAG(llm=mistral_client)
FILE_PATH = "data/document_test_rag.pdf"

@pytest.fixture
def vector_store():
    vector_store = LANGCHAIN_RAG.create_vector_store()
    yield vector_store
    vector_store.delete_collection()

def test_init():
    assert isinstance(LANGCHAIN_RAG, LangChainRAG) 
    assert isinstance(LANGCHAIN_RAG.embeddings_function, HuggingFaceEmbeddings)
    assert isinstance(LANGCHAIN_RAG.llm, ChatMistralAI)

def test_create_vector_store(vector_store):
    assert isinstance(vector_store, Chroma)

def test_chunk():   
    chunks = LANGCHAIN_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    assert len(documents) > 1
    for document in documents : 
        assert document
        assert len(document.page_content)<=500

def test_add_documents(vector_store):
    chunks = LANGCHAIN_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    ids = chunks['ids']
    LANGCHAIN_RAG.add_documents(vector_store, documents, ids)
    vector_store_documents = vector_store.get()['documents']
    vector_store_ids = vector_store.get()['ids']
    assert vector_store_documents == [doc.page_content for doc in documents]
    assert vector_store_ids == ids

def test_retrieve(vector_store):
    chunks = LANGCHAIN_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    ids = chunks['ids']
    LANGCHAIN_RAG.add_documents(vector_store, documents, ids)
    retrieved_documents = LANGCHAIN_RAG.retrieve(vector_store, "Quels résultats le programme a-t-il obtenus en septembre 2026 ?",k=3)
    assert len(retrieved_documents) == 3
    assert all(isinstance(doc, Document) for doc in retrieved_documents)

def test_answer(vector_store):
    chunks = LANGCHAIN_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    ids = chunks['ids']
    LANGCHAIN_RAG.add_documents(vector_store, documents, ids)
    context = LANGCHAIN_RAG.retrieve(vector_store, "Quels résultats le programme a-t-il obtenus en septembre 2026 ?",k=3)
    answer = LANGCHAIN_RAG.answer("Quels résultats le programme a-t-il obtenus en septembre 2026 ?", context)
    assert answer.strip()