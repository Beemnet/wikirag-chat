import os
import streamlit as st

from ingestion_pipeline import IngestionPipeline
from retrieval_pipeline import RetrievalPipeline
from answer_generation import AnswerGenerator
from history_aware_generation import HistoryAwareGenerator

from langchain_ollama import ChatOllama


## page config
st.set_page_config(page_title="Rag Demo", layout="centered")

st.title("RAG pipeline with Ingestion, Retrieval, and Answer Generation")
st.write("Single-page RAG with optional ingestion and history-aware queries.")

PERSIST_DIR = "db/chroma"

## session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = os.path.exists(PERSIST_DIR)


## ingestion
st.subheader("Ingestion Pipeline")

if st.button("Run Ingestion (Optional)"):
    with st.spinner("Running Ingestion ..."):
        pipeline = IngestionPipeline()
        pipeline.run()
        st.session_state.rag_ready = True
    st.success("Vector database ready.")

if not st.session_state.rag_ready:
    st.warning("Vector database not found. Run ingestion first.")


## retrieval 
st.subheader("Ask a question.")

query = st.text_input(
    "Question", 
    placeholder="How much did Microsoft pay for Github?"
)

## answer generation
if st.button("Send") and query:
    if not st.session_state.rag_ready:
        st.error("Vector database not ready, run Ingestion Pipeline first.")
    else: 
        with st.spinner("Processing ..."):
            ## history aware re-writing
            model = ChatOllama(model='glm-4.6:cloud')

            # update chat history with question
            st.session_state.chat_history.append(('human', query))

            if st.session_state.chat_history: 
                rewrite_messages = [
                    ("system",
                     "Given the chat history, rewrite the new question to be standalone and searchable."),
                    *st.session_state.chat_history,
                    ("human", f"New question: {query}")
                ]
                rewritten = model.invoke(rewrite_messages).content.strip()
            else: 
                rewritten = query
            
            # answer
            retriever = RetrievalPipeline()
            answer_gen = AnswerGenerator(retriever=retriever)
            answer = answer_gen.generate_answer(rewritten)

            # update chat history with answer
            st.session_state.chat_history.append(('ai', answer))

            # display
            st.markdown("### Answer")
            st.write(answer)

## chat history: 
if st.session_state.chat_history: 
    st.subheader("Conversation history")
    for role, content in st.session_state.chat_history:
        if role == "human":
            with st.chat_message("user"):
                st.markdown(f"**You:** {content}")
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
