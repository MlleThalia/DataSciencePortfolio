
from comparative_rag.scripts.basic_rag import BasicRAG
from comparative_rag.models.tracing import Run
from comparative_rag.ui.utils.utils import build_llm_client, create_run

def run_basic_rag(file : str, question : str):

    llm = build_llm_client()
    run = Run(**create_run())
    basic_rag = BasicRAG(llm)
    collection = basic_rag.create_collection()
    chunks = basic_rag.chunk(file)
    basic_rag.add_documents(collection, chunks['documents'], chunks['ids'])
    context = basic_rag.retrieve(collection, question)
    answer = basic_rag.answer(question, context, "", run)
    return answer