from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import logging
import pdfplumber

logger = logging.getLogger(__name__)

class LlamaIndexRAG():
    def __init__(self, llm : MistralAI, embeddings_model_name : str = "all-MiniLM-L6-v2"):
        self.embeddings_function = HuggingFaceEmbedding(model_name= embeddings_model_name)
        self.llm = llm

    def create_index(self, documents : list[Document])-> VectorStoreIndex | None:
        """Returns a vector store index."""
        try : 
            index = VectorStoreIndex.from_documents(documents, embed_model = self.embeddings_function)
            logger.info("Index created with sucess!")
            return index
        except Exception as e:
            logger.error("Error when creating the vector store : %s", e)

    def chunk(self, file_path : str, chunk_size = 500, chunk_overlap = 100):
        """Reads the PDF file and chunks the text in many chunks with SentenceSplitter."""
        try : 
            with pdfplumber.open(file_path) as pdf :
                text = ""
                for page in pdf.pages : 
                    text+= page.extract_text() + "\n\n"
            
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")

            splitter = SentenceSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            chunks = splitter.split_text(text)

            len_chunks = len(chunks)
            documents = [Document(text=chunk) for chunk in chunks]
            ids = [f"id_{i}" for i in range(1, len_chunks+1)]

            documents = {"ids": ids, "documents" : documents}

            return documents
        except Exception as e:
            logger.error("Error when chunking : %s", e)

    def answer(self, index : VectorStoreIndex, question : str) -> str | None :
        """Returns the llm answer using llama-index query engine."""
        try : 
            query_engine = index.as_query_engine(llm=self.llm)
            answer = query_engine.query(question)
            return answer.response
        except Exception as e :
            logger.error("Error when answering the question : %s", e)