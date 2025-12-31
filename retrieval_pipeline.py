from typing import List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class RetrievalPipeline: #retrieval from existing vector store
    def __init__(
            self,
            persist_directory: str = "db/chroma",
            embedding_model: str = "llama2",
            search_type: str = "similarity",
            k: int = 5,
            score_threshold: Optional[float] = 0.5

    ):
        
        self.persist_directory = persist_directory
        self.search_type = search_type
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store: Optional[Chroma] = None
        self.retriever = None

        self._load_vector_store()
        self._configure_retriever()


    def _load_vector_store(self):
        self.vector_store = Chroma(
            persist_directory = self.persist_directory,
            collection_metadata = {"hnsw:space": "cosine"}
        )
    

    def _configure_retriever(self) -> None:
        search_kwargs = {"k": self.k}
        if self.search_type == "similarity_score_threshold":
            if self.score_threshold:
                search_kwargs["score_threshold"] = self.score_threshold
            else: 
                raise ValueError("score_threshold must not be None.")
        
        self.retriever = self.vector_store.as_retriever(
            search_type = self.search_type,
            search_kwargs = search_kwargs
        )
        

    def retrieve(self, query:str) -> List[Document]:
        
        if not query: query = "How much did Microsoft pay for Github?"

        if not self.retriever: 
            raise RuntimeError("Retriever not initialized successfully.")
        return self.retriever.invoke(query)


if __name__ == "__main__":
    query = "How much did Microsoft pay for Github?"
    retrieval = RetrievalPipeline(search_type="similarity_score_threshold", score_threshold=0.5)

    relevant_docs = retrieval.retrieve(query=query)

    print(f"Query: {query}")
    for i, doc in enumerate(relevant_docs):
        print(f"Doc {i} with content: \n{doc.page_content}\n")

