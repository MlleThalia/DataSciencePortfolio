import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pytest

from comparative_rag.scripts.basic_rag import BasicRAG
from comparative_rag.ui.utils.utils import build_llm_client, create_run
from comparative_rag.llm.mistral_client import MistralClient
from comparative_rag.models.tracing import Run

mistral_client = build_llm_client()
BASIC_RAG = BasicRAG(llm=mistral_client)
FILE_PATH = "data/document_test_rag.pdf"

@pytest.fixture
def collection():
    collection = BASIC_RAG.create_collection()
    yield collection
    BASIC_RAG.client.delete_collection(collection.name)

def test_init():
    assert isinstance(BASIC_RAG, BasicRAG) 
    assert isinstance(BASIC_RAG.embeddings_function, SentenceTransformerEmbeddingFunction)
    assert isinstance(BASIC_RAG.client, chromadb.ClientAPI)
    assert isinstance(BASIC_RAG.llm, MistralClient)

def test_create_collection(collection):
    assert isinstance(collection, chromadb.Collection)

def test_chunk():   
    chunks = BASIC_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    assert len(documents) > 1
    for document in documents : 
        assert document
        assert len(document)<=500

def test_add_documents(collection):
    chunks = BASIC_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    ids = chunks['ids']
    BASIC_RAG.add_documents(collection, documents, ids)
    collection_documents = collection.get()['documents']
    collection_ids = collection.get()['ids']
    assert collection_documents == documents
    assert collection_ids == ids

def test_retrieve(collection):
    chunks = BASIC_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    ids = chunks['ids']
    BASIC_RAG.add_documents(collection, documents, ids)
    results = BASIC_RAG.retrieve(collection, "Quels résultats le programme a-t-il obtenus en septembre 2026 ?",k=3)
    retrieved_documents = results['documents'][0]
    assert len(retrieved_documents) == 3
    assert all(retrieved_documents)

def test_answer(collection):
    chunks = BASIC_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    ids = chunks['ids']
    BASIC_RAG.add_documents(collection, documents, ids)
    context = BASIC_RAG.retrieve(collection, "Quels résultats le programme a-t-il obtenus en septembre 2026 ?",k=3)
    answer = BASIC_RAG.answer("Quels résultats le programme a-t-il obtenus en septembre 2026 ?", context, "", Run(**create_run()))
    assert answer.strip()