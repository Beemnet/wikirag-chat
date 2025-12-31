import os
import unicodedata
from typing import List, Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter #for the chunks
from langchain_chroma import Chroma #as the vector db 
from langchain_ollama import OllamaEmbeddings 
from langchain_core.documents import Document

from dotenv import load_dotenv

class IngestionPipeline: # handles loading, chunking, embedding, vector store persistance
    def __init__(
            self, 
            docs_path: str = "docs",
            persist_directory: str = "db/chroma",
            chunk_size: int = 1000,
            chunk_overlap : int = 5,
            embedding_model : str = "llama2",

    ):
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OllamaEmbeddings(model=embedding_model)

        self.documents: Optional[List[Document]] = None
        self.chunks: Optional[List[Document]] = None
        self.vector_store: Optional[Chroma] = None


    def load_documents(self) -> List[Document]:
        print(f"\nLoading docs form {self.docs_path}.")

        if not os.path.exists(self.docs_path): 
            raise FileNotFoundError(f"Directory {self.docs_path} does not exist.")
        
        loader = DirectoryLoader(
            path = self.docs_path,
            glob = "*.txt",
            loader_cls = TextLoader
        )

        documents = loader.load()

        if len(documents) == 0:
            raise FileNotFoundError(f"No .txt files found in {self.docs_path}.")
        
        # for i, doc in enumerate(documents[:2]):
        #     print(f"\nDocument {i+1} out of {len(documents)}:")
        #     print(f"  Source: {doc.metadata['source']}")
        #     print(f"  Content length: {len(doc.page_content)} characters")
        #     print(f"  Content preview: {doc.page_content[:100]}...")
        #     print(f"  metadata: {doc.metadata}")      

        self.documents = documents  
        return self.documents

    def split_documents(self) -> List[Document]:
        if self.documents is None: 
            raise RuntimeError("Documents not loaded. Check load_documents() has been called and run successfully.")
        
        print(f"\nSplitting {len(self.documents)} documents into chunks.")

        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = text_splitter.split_documents(self.documents)

        # if chunks:
        #     for i, chunk in enumerate(chunks[:5]):
        #         print(f"\n--- Chunk {i+1} out of {len(chunks)} ---")
        #         print(f"Source: {chunk.metadata['source']}")
        #         print(f"Length: {len(chunk.page_content)} characters")
        #         print(f"Content:")
        #         print(chunk.page_content)
        #         print("-" * 50)            

        self.chunks = chunks
        return self.chunks


    def _normalize_chunks(self) -> None:
        if not self.chunks:
            return
        for doc in self.chunks:
            doc.page_content = unicodedata.normalize("NFKC", doc.page_content)


    def create_vector_store(self) -> Chroma:
        print("Creating embeddings and storing Chroma vectordb")

        if os.path.exists(self.persist_directory):
            print(
                f"Vector store already exists at directory {self.persist_directory}, "
                "not processing documents."
            )
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
                # embedding = self.embeddings # commented out as ollama accepts this implicitly
            )
            return self.vector_store        
        if not self.chunks: 
            raise RuntimeError("No chunks available to index.")
        
        # Normalize text to avoid Unicode decode issues
        self._normalize_chunks()

        self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
        print(f"Vector store created and stored at directory {self.persist_directory}")

        return self.vector_store

    # Full ingestion pipeline
    def run(self) -> Chroma:
        self.load_documents()
        self.split_documents()
        return self.create_vector_store()

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()

