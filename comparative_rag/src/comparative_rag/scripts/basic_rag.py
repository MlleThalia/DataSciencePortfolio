import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging
from dotenv import load_dotenv
from pathlib import Path
import os
import json
import time
import pdfplumber

from comparative_rag.llm.mistral_client import MistralClient
from comparative_rag.models.tracing import LlmCall, Run, Trace

logger = logging.getLogger(__name__)
load_dotenv()

OUTPUT_DIR = Path("traces/")

MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))

class BasicRAG():
    def __init__(self, llm : MistralClient, embeddings_model_name : str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.embeddings_function = SentenceTransformerEmbeddingFunction(model_name = embeddings_model_name)
        self.llm = llm

    def create_collection(self, collection_name : str = "rag_documents")-> chromadb.Collection | None:
        """Returns a Chromadb Collection."""
        try : 
            collection = self.client.create_collection(name = collection_name,
                                                        embedding_function=self.embeddings_function)
            logger.info("Collection %s created with sucess!", collection.name)
            return collection
        except Exception as e:
            logger.error("Error when creating the collection : %s", e)

    def chunk(self, file_path : str, chunk_size : int = 500, chunk_overlap : int = 100)-> dict | None:
        """Reads the PDF file and chunks the text in many chunks."""
        try : 
            with pdfplumber.open(file_path) as pdf :
                text = ""
                for page in pdf.pages : 
                    text+= page.extract_text() + "\n\n"
            
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size.")

            chunks = []
            start = 0

            while start < len(text):
                end = start + chunk_size

                chunk = text[start:end]
                chunks.append(chunk)

                start += chunk_size - chunk_overlap

            len_chunks = len(chunks)
            ids = [f"id_{i}" for i in range(1, len_chunks+1)]

            documents = {"ids": ids, "documents" : chunks}

            return documents
        except Exception as e:
            logger.error("Error when chunking : %s", e)
    
    def add_documents(self, collection : chromadb.Collection, documents : list, ids : list) -> None:
        """Adds documents to a collection."""
        try : 
            collection.add(ids=ids,
                        documents=documents)
            logging.info("Documents added to the collection with sucess!")
        except Exception as e :
            logger.error("Error when adding documents to the collection : %s", e)

    def retrieve(self, collection : chromadb.Collection, query : str, k=3) -> dict | None:
        """Returns the top 3 vectors similar to the query."""
        try : 
            results = collection.query(query_texts=[query], n_results=k)
            logging.info("Retrieving done with sucess!")
            return results
        except Exception as e :
            logger.error("Error when retrieving : %s", e)

    def answer(self, question : str, context : str, system_prompt : str, run : Run) -> str | None :
        """Returns the llm answer based on the context."""
        try : 
            prompt = self._build_prompt(question, context)
            llm_calls = []

            for attempt_number in range(1, MAX_RETRY+2):

                response = self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    attempt_number=attempt_number,
                )

                logger.info(f"Answering the question | {attempt_number} attempt.")
                llm_response = response["llm_response"]
                llm_trace = response["trace"]
                llm_calls.append(llm_trace)

                if not llm_trace.errors :
                    break

                if "http_error" in llm_trace.errors:
                    time.sleep(1)
                    continue
            
            trace = Trace(run=run, llm_calls=llm_calls)
            self._save_json(trace, f"trace_{run.run_id}.json")
            return llm_response
        except Exception as e :
            logger.error("Error when answering the question : %s", e)
    
    def _build_prompt(self, question : str, context: str) -> str:
        """Returns the formatted prompt."""

        return f"""Answer the question using exclusively the provided context.

        If the context does not contain enough information,
        explicitly state that you cannot answer the question.

        Do not use external knowledge.

        Question:
        {question}

        Context:
        {context}
        """

    def _save_json(self, data, filename: str):
        """Saves the trace."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        path = OUTPUT_DIR / filename

        if hasattr(data, "model_dump"):
            data = data.model_dump(mode="json")

        path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"✓ Saved {path}")