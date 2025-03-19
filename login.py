import streamlit as st
from dataclasses import dataclass
import os
import pyttsx3
import speech_recognition as sr
import time
import threading
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt_tab')
from typing import Literal
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter

VALID_USERNAME = "user"
VALID_PASSWORD = "password123"

def login_page():
    st.title("Login to AI Interview System")
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Login successful! Redirecting to the interview system...")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

def main_app():
    def speak(text):
        def tts():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        thread = threading.Thread(target=tts)
        thread.start()

    st.title("AI Interview System")

    position = st.selectbox("Select the position:", ["Data Analyst", "Software Engineer", "Cyber Security", "Web Development"])
    resume = st.file_uploader("Upload your resume", type=["pdf", "txt"])
    auto_play = st.checkbox("Let AI interviewer speak!")
    voice_input = st.checkbox("Use voice input for answers")

    @dataclass
    class Message:
        origin: Literal["human", "ai"]
        message: str

    def process_resume(resume):
        nltk.download('punkt_tab')
        text = ""
        if resume.type == "application/pdf":
            pdf_reader = PdfReader(resume)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        else:
            text = resume.read().decode("utf-8")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(texts, embeddings)

    def initialize_session():
        if 'docsearch' not in st.session_state:
            st.session_state.docsearch = process_resume(resume)
        if 'retriever' not in st.session_state:
            st.session_state.retriever = st.session_state.docsearch.as_retriever(search_type="similarity")
        if "resume_history" not in st.session_state:
            st.session_state.resume_history = [Message("ai", "Tell me about yourself")]
        if "resume_memory" not in st.session_state:
            st.session_state.resume_memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        if "resume_screen" not in st.session_state:
            groq_api_key = os.getenv("GROQ_API_KEY", "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")
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
            PROMPT = PromptTemplate(
                input_variables=["history", "input"],
                template="""
                I am an AI interviewer conducting a structured technical interview. I will ask short and precise questions based on the candidate’s resume, followed by DSA and coding questions.

### Interview Structure:
1. **Resume-Based Questions:** Ask 2-3 line questions focused on key skills, projects, and experience.
2. **DSA Questions:** Ask three concise DSA questions (easy, medium, hard).
3. **Coding Questions:** Ask 1-2 short coding problems relevant to the candidate's job role.

### Interview Flow:
- Start by asking the candidate to introduce themselves.
- Ask skill-based questions, keeping them direct and efficient.
- Transition smoothly into DSA and coding questions.
- Ensure questions are engaging, relevant, and not overly lengthy.
-Give only review and rating of answer. dont give updated code

### Output Format:
- Use 2-3 line questions.
- Keep follow-ups precise and goal-oriented.
- If the candidate struggles, provide short hints instead of direct answers.

Let's start the interview. Ask the candidate to introduce themselves.
                
Current Conversation:
{history}

Candidate: {input}
                """
            )
            st.session_state.resume_screen = ConversationChain(
                llm=llm,
                memory=st.session_state.resume_memory,
                prompt=PROMPT,
                verbose=True
            )
            st.session_state.start_time = time.time()
            st.session_state.question_count = 0

    def transcribe_audio():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            st.info("🔍 Processing your speech...")
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Transcribed: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand.")
            return ""

    def query_with_retry(chain, user_input, retries=3, delay=5):
        for _ in range(retries):
            try:
                return chain.run(input=user_input)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise e
        return "Request failed due to repeated rate limit errors."

    def answer_callback():
        if st.session_state.question_count >= 12:
            st.write("Thank you for completing the interview!")
            return
        human_answer = st.session_state.get("answer", "")
        st.session_state.resume_history.append(Message("human", human_answer))
        with st.spinner("AI thinking..."):
            ai_response = query_with_retry(st.session_state.resume_screen, human_answer)
        st.session_state.resume_history.append(Message("ai", ai_response))
        st.session_state.question_count += 1
        if auto_play:
            speak(ai_response)

    if position and resume:
        initialize_session()
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.resume_history):
                with st.chat_message(msg.origin):
                    st.write(msg.message)
        input_container = st.container()
        with input_container:
            if voice_input:
                if st.button("🎤 Answer with Voice"):
                    transcribed_text = transcribe_audio()
                    if transcribed_text:
                        st.session_state["answer"] = transcribed_text
                        answer_callback()
                        st.rerun()
            user_input = st.text_area("Your Answer:", key="user_input", help="Enter your response here...", height=200)
            if st.button("Submit Text Answer"):
                if user_input.strip():
                    st.session_state["answer"] = user_input
                    answer_callback()
                    st.rerun()
                else:
                    st.warning("Please enter your answer before submitting.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    login_page()
else:
    main_app()
