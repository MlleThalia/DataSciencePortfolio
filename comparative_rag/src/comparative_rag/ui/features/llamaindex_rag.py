
from comparative_rag.scripts.llamaindex_rag import LlamaIndexRAG
from comparative_rag.ui.utils.utils import mistral_ai

def run_llamaindex_rag(file : str, question : str):

    llm = mistral_ai()
    langchain_rag = LlamaIndexRAG(llm)
    chunks = langchain_rag.chunk(file)
    index = langchain_rag.create_index(chunks['documents'])
    answer = langchain_rag.answer(index, question)
    return answer