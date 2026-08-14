# ==========================================================
# OLLAMA | FAISS | STREAMLIT | VOICE → VOICE
# WITH PERSISTENT MULTI-DOCUMENT INDEXING
# ==========================================================

import os, re, base64, tempfile, hashlib
import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav

from dotenv import load_dotenv
from gtts import gTTS
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Doc Chatbot", layout="centered")
st.title("📄 Document Chatbot")

# ==========================================================
# DIRECTORIES
# ==========================================================
UPLOAD_DIR = "data/uploads"
INDEX_DIR = "data/indexes"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ==========================================================
# SESSION STATE
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_index" not in st.session_state:
    st.session_state.active_index = None

# ==========================================================
# LANGUAGES
# ==========================================================
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Arabic": "ar"
}

# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

if not OLLAMA_URL or not OLLAMA_MODEL:
    st.error("❌ Missing Ollama config")
    st.stop()

# ==========================================================
# MODELS
# ==========================================================
@st.cache_resource
def load_models():
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.2)
    whisper = WhisperModel("small", compute_type="int8")
    return embed, llm, whisper

embedding_model, llm, whisper_model = load_models()

# ==========================================================
# PROMPT
# ==========================================================
PROMPT = PromptTemplate.from_template("""
Answer using only the context below.

Context:
{context}

Question:
{input}
""")

# ==========================================================
# UTILS
# ==========================================================
def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def clean_text(t):
    return t.strip().replace("\n", " ")

def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(f.name)
    return f.name

def record_audio(sr=16000):
    audio = sd.rec(int(12 * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(f.name, sr, audio)
    return f.name

def transcribe(path, lang):
    segments, _ = whisper_model.transcribe(
        path,
        language=lang,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=600)
    )
    return " ".join(seg.text for seg in segments).strip()

def load_or_create_index(file_path, index_path):
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    loader = PyMuPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
    docs = loader.load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    ).split_documents(docs)

    index = FAISS.from_documents(
        [Document(page_content=clean_text(c.page_content)) for c in chunks],
        embedding_model
    )
    index.save_local(index_path)
    return index

# ==========================================================
# FILE UPLOAD
# ==========================================================
uploaded = st.file_uploader("Upload PDF or TXT", ["pdf", "txt"])

if uploaded:
    data = uploaded.getbuffer()
    h = file_hash(data)

    file_path = os.path.join(UPLOAD_DIR, f"{h}_{uploaded.name}")
    index_path = os.path.join(INDEX_DIR, h)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(data)

    with st.spinner("📚 Preparing document..."):
        index = load_or_create_index(file_path, index_path)
        st.session_state.active_index = index

    st.success("✅ Document ready (cached)")

# ==========================================================
# CHAT HISTORY
# ==========================================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ==========================================================
# CHAT INPUT BAR
# ==========================================================
st.markdown("---")

col_lang, col_input, col_mic = st.columns([1.2, 6, 0.8])

with col_lang:
    selected_lang = st.selectbox("🌐", list(LANGUAGES.keys()), label_visibility="collapsed")
    lang_code = LANGUAGES[selected_lang]

with col_input:
    prompt = st.chat_input("Ask your document...")


with col_mic:
    mic_clicked = st.button("🎙️", help="Speak", use_container_width=True)

# ==========================================================
# VOICE INPUT
# ==========================================================
if mic_clicked:
    with st.spinner("Listening..."):
        audio_path = record_audio()
        spoken = transcribe(audio_path, lang_code)

        if spoken:
            prompt = spoken
            st.session_state.input_mode = "voice"   # 🔥 KEY LINE
        else:
            st.warning("Could not understand audio")


# ==========================================================
# TEXT INPUT MODE
# ==========================================================
if prompt and not mic_clicked:
    st.session_state.input_mode = "text"
# ==========================================================
# PROCESS QUERY
# ==========================================================
if prompt and st.session_state.active_index:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            q_en = GoogleTranslator(source="auto", target="en").translate(prompt)

            chain = create_retrieval_chain(
                st.session_state.active_index.as_retriever(search_kwargs={"k": 3}),
                create_stuff_documents_chain(llm, PROMPT)
            )

            res = chain.invoke({"input": q_en})
            answer_en = res.get("answer", "")

            answer = (
                GoogleTranslator(source="en", target=lang_code).translate(answer_en)
                if lang_code != "en"
                else answer_en
            )

            # ✅ Always show text
            st.markdown(answer)

            # 🔊 Speak ONLY if input was voice
            if st.session_state.input_mode == "voice":
                st.audio(speak(answer, lang_code), autoplay=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )