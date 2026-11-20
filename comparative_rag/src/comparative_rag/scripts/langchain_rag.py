from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
import logging
import pdfplumber

logger = logging.getLogger(__name__)

class LangChainRAG():
    def __init__(self, llm : ChatMistralAI, embeddings_model_name : str = "all-MiniLM-L6-v2"):
        self.embeddings_function = HuggingFaceEmbeddings(model_name = embeddings_model_name)
        self.llm = llm

    def create_vector_store(self, vector_store_name : str = "rag_documents")-> Chroma | None:
        """Returns a Chromadb vector store."""
        try : 
            vector_store = Chroma(collection_name = vector_store_name,
                                embedding_function=self.embeddings_function)
            logger.info("Vector store %s created with sucess!", vector_store_name)
            return vector_store
        except Exception as e:
            logger.error("Error when creating the vector store : %s", e)

    def chunk(self, file_path : str, chunk_size = 500, chunk_overlap = 100):
        """Reads the PDF file and chunks the text in many chunks with RecursiveCharacterTextSplitter."""
        try : 
            with pdfplumber.open(file_path) as pdf :
                text = ""
                for page in pdf.pages : 
                    text+= page.extract_text() + "\n\n"
            
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")

            splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            chunks = splitter.split_text(text)

            len_chunks = len(chunks)
            documents = [Document(page_content=chunk) for chunk in chunks]
            ids = [f"id_{i}" for i in range(1, len_chunks+1)]

            documents = {"ids": ids, "documents" : documents}

            return documents
        except Exception as e:
            logger.error("Error when chunking : %s", e)
    
    def add_documents(self, vector_store : Chroma, documents : list, ids : list) -> None:
        """Adds documents to the vector store."""
        try : 
            vector_store.add_documents(ids=ids,
                        documents=documents)
            logging.info("Documents added to the vector store with sucess!")
        except Exception as e :
            logger.error("Error when adding documents to the vector store : %s", e)

    def retrieve(self, vector_store : Chroma, query, k=3) -> list[Document] | None:
        """Returns the top 3 vectors similar to the query."""
        try : 
            results = vector_store.similarity_search(query=query, k=k)
            logging.info("Retrieving done with sucess!")
            return results
        except Exception as e :
            logger.error("Error when retrieving : %s", e)

    def answer(self, question : str, context : str) -> str | None :
        """Returns the llm answer based on the context."""
        try : 
            prompt = self._build_prompt()
            llm_chain = prompt | self.llm | StrOutputParser()
            answer = llm_chain.invoke({"question":question, "context": context})
            return answer
        except Exception as e :
            logger.error("Error when answering the question : %s", e)
    
    def _build_prompt(self) -> PromptTemplate:
        """Returns a prompt template."""

        template = """Answer the question using exclusively the provided context.

        If the context does not contain enough information,
        explicitly state that you cannot answer the question.

        Do not use external knowledge.

        Question:
        {question}

        Context:
        {context}
        """

        return PromptTemplate(template=template, input_variables=["question", "context"])