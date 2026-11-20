import gradio as gr
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

from comparative_rag.ui.features.basic_rag import run_basic_rag
from comparative_rag.ui.features.langchain_rag import run_langchain_rag
from comparative_rag.ui.features.llamaindex_rag import run_llamaindex_rag

with gr.Blocks(title="Question - Answering") as demo:
    gr.Markdown("# Question - Answering")

    gr.Markdown("## Files")

    file = gr.File(label="Importer le PDF.",
                    file_count='single',
                    file_types=['.pdf'])
    
    gr.Markdown("## Question")

    question = gr.Textbox(label="Ask a question.", type='text')

    with gr.Row():
        basic_btn = gr.Button("Basic RAG", variant='primary')
        langchain_btn = gr.Button("LangChain", variant='primary')
        llamaindex_btn = gr.Button("LlamaIndex", variant='primary')

    gr.Markdown("## Answer")
    answer = gr.Markdown(label="Réponse")

    basic_btn.click(
    fn=run_basic_rag,
    inputs=[file, question],
    outputs=answer
    )

    langchain_btn.click(
        fn=run_langchain_rag,
        inputs=[file, question],
        outputs=answer
    )

    llamaindex_btn.click(
        fn=run_llamaindex_rag,
        inputs=[file, question],
        outputs=answer
    )


if __name__ == "__main__" :
    demo.launch()