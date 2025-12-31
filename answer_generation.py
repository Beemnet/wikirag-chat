from typing import List, Optional

from retrieval_pipeline import RetrievalPipeline
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

# import chat ollama

# from langchain_core.messages import HumanMessage, SystemMessage

class AnswerGenerator: 
    def __init__(
            self,
            retriever: Optional[RetrievalPipeline] = None,
            chat_model: str = "glm-4.6:cloud",
            system_prompt: str = "You are an informational assistant Use the given text to answer the query asked.", 

        ):
          
        self.retriever = retriever or RetrievalPipeline()
        self.chat_model = chat_model
        self.system_prompt = system_prompt

        self.relevant_documents: Optional[List[Document]] = None
        self.processed_query: Optional[str] = None

    def _retrieve_documents(self, query:str, search_type:str="similarity_score_threshold", score_threshold:int=0.3):

        if not query: query = "How much did Microsoft pay for Github?"
        
        self.retriever = RetrievalPipeline(search_type=search_type, score_threshold=score_threshold)

        self.relevant_docs = self.retriever.retrieve(query)

    
    def _process_query(self, query: str) -> None:

        if not query: query = "How much did Microsoft pay for Github?"

        self.processed_query = f"""Based on the following documents, please answer this question: {query}

        Documents: 
        {chr(10).join(f"[doc] - {doc.page_content}" for doc in self.relevant_docs)}

        provide a clear response, if you don't have enough information, respond "I do not have enough information"
        """

    def generate_answer(self, query:str = None) -> str:

        if not query: query = "How much did Microsoft pay for Github?"
        print(f"Answering question: {query}")
        self._retrieve_documents(query=query)
        self._process_query(query=query)

        self.model = ChatOllama(model=self.chat_model)


        messages = [
            ("system", self.system_prompt), 
            ("human", self.processed_query)
        ]

        result = self.model.invoke(messages)

        print(f"Response: {result.content}")
        return result.content
    

if __name__ == "__main__" : 
    answer_generator = AnswerGenerator()
    print(answer_generator.generate_answer())