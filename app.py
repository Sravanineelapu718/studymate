import os
import streamlit as st
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.schema import Document
from prompts import SYSTEM_PROMPT_QA, SYSTEM_PROMPT_QUIZ, SYSTEM_PROMPT_ELI5
from streamlit_chat import message

load_dotenv()

st.set_page_config(page_title="📚 StudyMate AI Assistant", layout="wide")
st.title("📘 StudyMate – PDF-Based Learning Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
user_query = st.text_input("Ask a question from your uploaded materials")
gen_quiz = st.button("🔍 Generate Quiz Questions")
eli5 = st.button("🌱 Explain Like I'm 5")

# === Load and split PDFs ===
docs = []
if uploaded:
    for pdf in uploaded:
        loader = PyPDFLoader(pdf)
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = pdf.name
        docs.extend(pages)

if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAI.embeddings_api_key = os.getenv("OPENAI_API_KEY")
    db = OpenAI(temperature=0)

    if user_query:
        chain = RetrievalQA.from_chain_type(llm=db, chain_type="stuff", retriever=None)
        answer = chain.run({"question": user_query, "documents": chunks})

        message("You: " + user_query, is_user=True)
        message("StudyMate: " + answer)

        if eli5:
            el_chain = load_summarize_chain(llm=db, chain_type="stuff")
            simplified = el_chain.run([Document(page_content=SYSTEM_PROMPT_ELI5.format(answer=answer))])
            message("ELI5: " + simplified)

        st.download_button("📅 Download Answer", answer, file_name="answer.txt")

    if gen_quiz:
        quiz_chain = load_summarize_chain(llm=db, chain_type="stuff")
        quiz = quiz_chain.run([Document(page_content=SYSTEM_PROMPT_QUIZ)] + chunks)
        message("Quiz Questions: \n" + quiz)
        st.download_button("📅 Download Quiz", quiz, file_name="quiz.txt")
