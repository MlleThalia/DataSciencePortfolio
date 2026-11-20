
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core import VectorStoreIndex
from llama_index.core.utils import get_tokenizer

from comparative_rag.scripts.llamaindex_rag import LlamaIndexRAG
from comparative_rag.ui.utils.utils import mistral_ai

mistral_client = mistral_ai()
LLAMAINDEX_RAG = LlamaIndexRAG(llm=mistral_client)
FILE_PATH = "data/document_test_rag.pdf"

def test_init():
    assert isinstance(LLAMAINDEX_RAG, LlamaIndexRAG) 
    assert isinstance(LLAMAINDEX_RAG.embeddings_function, HuggingFaceEmbedding)
    assert isinstance(LLAMAINDEX_RAG.llm, MistralAI)

def test_chunk():   
    chunks = LLAMAINDEX_RAG.chunk(FILE_PATH)
    documents = chunks['documents']
    assert len(documents) > 1
    tokenizer = get_tokenizer()
    for document in documents : 
        assert document
        assert len(tokenizer(document.text))<=500

def test_create_index():
    chunks = LLAMAINDEX_RAG.chunk(FILE_PATH)
    index = LLAMAINDEX_RAG.create_index(chunks['documents'])
    assert isinstance(index, VectorStoreIndex)

def test_answer():
    chunks = LLAMAINDEX_RAG.chunk(FILE_PATH)
    index = LLAMAINDEX_RAG.create_index(chunks['documents'])
    answer = LLAMAINDEX_RAG.answer(index, "Quels résultats le programme a-t-il obtenus en septembre 2026 ?")
    assert answer.strip()