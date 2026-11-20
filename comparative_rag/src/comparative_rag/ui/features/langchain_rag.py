
from comparative_rag.scripts.langchain_rag import LangChainRAG
from comparative_rag.ui.utils.utils import chat_mistral_ai

def run_langchain_rag(file : str, question : str):

    llm = chat_mistral_ai()
    langchain_rag = LangChainRAG(llm)
    vector_store = langchain_rag.create_vector_store()
    chunks = langchain_rag.chunk(file)
    langchain_rag.add_documents(vector_store, chunks['documents'], chunks['ids'])
    context = langchain_rag.retrieve(vector_store, question)
    answer = langchain_rag.answer(question, context)
    return answer