from typing import List, Optional

from retrieval_pipeline import RetrievalPipeline

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama



load_dotenv()


class HistoryAwareGenerator: 
    def __init__(
            self, 
            chat_model: str = "glm-4.6:cloud",
            retriever: Optional[RetrievalPipeline] = None
    ):
        self.retriever = retriever or RetrievalPipeline()
        self.model = ChatOllama(model=chat_model)
        self.rewritten_query: str = None
        self.chat_history: List[AIMessage, HumanMessage] = []


    def _rewrite_question(self, user_question:str) -> str:
        if not self.chat_history:
            self.rewritten_query= user_question
        
        messages = [
                ("system", "Given the chat history, rewrite the new question to be standalone and searchable and just return the rewritten question."),
                *self.chat_history,
                ("human", f"New Question: {user_question}")
        ]

        rewritten = self.model.invoke(messages)
        self.rewritten_query = rewritten.content.strip()


    def ask(self, user_question) -> str:

        # retrieve info 
        self._rewrite_question(user_question)
        print(f"Asking question: {self.rewritten_query}")
        relevant_docs = self.retriever.retrieve(self.rewritten_query)

        # generate answer
        combined_prompt = f"""based on the following documents, please answer this question: {self.rewritten_query}

        Documents: 
        {chr(10).join(f"[doc] - {doc.page_content}" for doc in relevant_docs)}

        provide a clear response, if you don't have enough information, respond "I do not have enough information"
        """

        messages = [
            ('system', 'you are an informational assistant.'),
            ('human', combined_prompt)
        ]

        response = self.model.invoke(messages).content

        self.chat_history.append(HumanMessage(content=user_question))
        self.chat_history.append(AIMessage(content=response))
        return response


if __name__ == "__main__":
    chat = HistoryAwareGenerator()

    print(chat.ask("How much did Microsoft pay for GitHub?"))
    print(chat.ask("When did that happen?"))