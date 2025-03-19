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
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from typing import Literal
from groq import Groq

# Hardcoded credentials for demo purposes (replace with secure auth in production)
VALID_USERNAME = "user"
VALID_PASSWORD = "password123"

# Login Page
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
                st.rerun()  # Rerun to switch to main app
            else:
                st.error("Invalid username or password. Please try again.")

# Main Interview System (your provided code, unchanged)
def main_app():
    # Initialize text-to-speech engine
    def speak(text):
        """Convert AI's response to speech in a separate thread."""
        def tts():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        
        thread = threading.Thread(target=tts)
        thread.start()

    # UI Elements
    st.title("AI Interview System")

    position = st.selectbox("Select the position:", ["Data Analyst", "Software Engineer", "Cyber Security", "Web Development"])
    resume = st.file_uploader("Upload your resume", type=["pdf", "txt"])
    auto_play = st.checkbox("Let AI interviewer speak!")
    voice_input = st.checkbox("Use voice input for answers")

    @dataclass
    class Message:
        """Class to store interview history."""
        origin: Literal["human", "ai"]
        message: str

    def process_resume(resume):
        """Extracts text from PDF/TXT resume and converts it into embeddings."""
        nltk.download('punkt')
        
        text = ""
        if resume.type == "application/pdf":
            pdf_reader = PdfReader(resume)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        else:  # Process .txt files
            text = resume.read().decode("utf-8")
        
        text_splitter = NLTKTextSplitter()
        texts = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(texts, embeddings)

    def initialize_session():
        """Initialize session state variables."""
        if 'docsearch' not in st.session_state:
            st.session_state.docsearch = process_resume(resume)
        
        if 'retriever' not in st.session_state:
            st.session_state.retriever = st.session_state.docsearch.as_retriever(search_type="similarity")
        
        if "resume_history" not in st.session_state:
            st.session_state.resume_history = [Message("ai", "Tell me about yourself")]
        
        if "resume_memory" not in st.session_state:
            st.session_state.resume_memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        if "resume_screen" not in st.session_state:
            # Use environment variable for API key for better security
            # groq_api_key = os.getenv("GROQ_API_KEY", "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")
            client = Groq(api_key=st.secrets["auth_key"])
            llm = ChatGroq(
                groq_api_key=client,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )

            PROMPT = PromptTemplate(
                input_variables=["history", "input"],
                template="""I am an AI interviewer conducting a structured technical interview. I will ask short and precise questions based on the candidate’s resume, followed by DSA and coding questions.

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
        """Uses Google Speech Recognition to convert audio to text."""
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
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            st.error("Speech Recognition service unavailable.")
            return "Speech Recognition service unavailable."

    def query_with_retry(chain, user_input, retries=3, delay=5):
        """Handles rate limits by retrying if Groq API returns 429 error."""
        for _ in range(retries):
            try:
                return chain.run(input=user_input)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)  # Wait before retrying
                else:
                    raise e  # Raise other errors
        return "Request failed due to repeated rate limit errors."

    def answer_callback():
        """Handles user's answer, processes response using Groq LLM, and speaks AI's response."""
        if st.session_state.question_count >= 12:
            st.write("Thank you for completing the interview!")
            return
        
        human_answer = st.session_state.get("answer", "")
        
        st.session_state.resume_history.append(Message("human", human_answer))
        
        # Show spinner while getting AI response
        with st.spinner("AI thinking..."):
            ai_response = query_with_retry(st.session_state.resume_screen, human_answer)
        
        st.session_state.resume_history.append(Message("ai", ai_response))
        st.session_state.question_count += 1
        
        if auto_play:
            speak(ai_response)

    def generate_history_text():
        """Converts the interview history into a plain text string."""
        history_text = ""
        for msg in st.session_state.resume_history:
            role = "Candidate" if msg.origin == "human" else "Interviewer"
            history_text += f"{role}: {msg.message}\n\n"
        return history_text

    def rate_interview():
        """Sends the entire conversation to the LLM to get an overall rating and feedback."""
        # Build the rating prompt with conversation history
        history_text = generate_history_text()
        rating_prompt = f"""
    Please review the following interview conversation and rate the candidate's performance on a scale of 1-10.
    Provide a brief summary of strengths and areas for improvement.

    Interview Conversation:
    {history_text}
    """
        with st.spinner("Evaluating your interview performance..."):
            rating_feedback = query_with_retry(st.session_state.resume_screen, rating_prompt)
        return rating_feedback

    if position and resume:
        initialize_session()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            feedback_btn = st.button("Get Interview Feedback")
        with col2:
            guideline_btn = st.button("Show Interview Guideline")
        with col3:
            download_btn = st.button("Download Interview History")
        with col4:
            rate_btn = st.button("Rate My Interview")

        if guideline_btn:
            st.markdown("""Interview Guidelines:
    1. Be confident  
    2. Answer concisely  
    3. Provide examples""")

        if feedback_btn:
            st.markdown("AI Feedback: Your interview performance was good, but try to elaborate on past experiences.")
            st.download_button("Download Feedback", "AI Feedback", file_name="interview_feedback.txt")
            st.stop()
        
        if download_btn:
            history_text = generate_history_text()
            st.download_button("Download Interview History", history_text, file_name="interview_history.txt")
        
        if rate_btn:
            rating = rate_interview()
            st.markdown("### Interview Rating and Feedback")
            st.write(rating)
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.resume_history):
                with st.chat_message(msg.origin):
                    st.write(msg.message)

        # Input section
        input_container = st.container()
        with input_container:
            if voice_input:
                if st.button("🎤 Answer with Voice"):
                    transcribed_text = transcribe_audio()
                    if transcribed_text and transcribed_text not in ["Sorry, I couldn't understand.", "Speech Recognition service unavailable."]:
                        st.session_state["answer"] = transcribed_text
                        answer_callback()
                        st.rerun()
            
            # Always provide text input option
            user_input = st.text_input("Your Answer:", key="user_input")
            if st.button("Submit Text Answer"):
                if user_input.strip():
                    st.session_state["answer"] = user_input
                    answer_callback()
                    st.rerun()
                else:
                    st.warning("Please enter your answer before submitting.")

        st.write("### Notepad for Coding Questions")
        user_code = st.text_area("Write your code here...", key="user_code")

        if st.button("Submit Code"):
            if user_code.strip():
                evaluation_prompt = f"""
    Evaluate the following Python code in terms of correctness, efficiency, and best practices.
    Provide improvement suggestions if needed.

    Code:

    {user_code}
    """
                with st.spinner("Evaluating your code..."):
                    ai_feedback = query_with_retry(st.session_state.resume_screen, evaluation_prompt)
                
                st.write("✅ Your code has been submitted successfully!")
                st.write("📝 AI Review:")
                st.write(ai_feedback)

                # Add code feedback to the conversation history
                st.session_state.resume_history.append(Message("human", f"[CODE SUBMISSION]\n{user_code}"))
                st.session_state.resume_history.append(Message("ai", f"[CODE REVIEW]\n{ai_feedback}"))
                st.rerun()
            else:
                st.write("⚠ Please enter code before submitting.")

# Main execution logic
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
else:
    main_app()
