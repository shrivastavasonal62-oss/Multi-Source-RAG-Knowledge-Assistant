import os
import html
import tempfile
from io import BytesIO

import streamlit as st
import joblib
import numpy as np
import faiss
import whisper
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from streamlit_mic_recorder import mic_recorder

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3
MIN_SCORE = 0.20


def load_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


st.set_page_config(
    page_title="Multi-Source RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

load_css()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""


# -------------------- HEADER --------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 Multi-Source <span>RAG</span> Knowledge Assistant</div>
    <div class="subtitle">
        Chat with multiple PDFs, lecture transcripts, and voice input using
        Retrieval-Augmented Generation, FAISS vector search, Whisper, and Groq LLM.
    </div>
    <div>
        <span class="top-badge">📄 Multi-PDF RAG</span>
        <span class="top-badge">📌 Page Citations</span>
        <span class="top-badge">🎙️ Voice Input</span>
        <span class="top-badge">📥 PDF Export</span>
        <span class="top-badge">💬 Chat History</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")


encoder = load_encoder()


@st.cache_data
def load_lecture_vectors():
    try:
        art = joblib.load("vectors.joblib")
        texts = art["texts"]

        lecture_chunks = []
        for i, text in enumerate(texts):
            lecture_chunks.append({
                "text": text,
                "source": "Lecture Transcript",
                "page": "N/A",
                "type": "Lecture",
                "chunk_id": i + 1
            })

        return lecture_chunks

    except Exception:
        return []


# -------------------- FUNCTIONS --------------------
def transcribe_audio(audio_bytes):
    whisper_model = load_whisper_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    result = whisper_model.transcribe(temp_audio_path)
    return result["text"].strip()


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def extract_pdf_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    pdf_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if not page_text:
            continue

        page_chunks = chunk_text(page_text)

        for chunk in page_chunks:
            pdf_chunks.append({
                "text": chunk,
                "source": pdf_file.name,
                "page": page_num,
                "type": "PDF",
                "chunk_id": len(pdf_chunks) + 1
            })

    return pdf_chunks


@st.cache_resource
def cached_build_index(chunk_texts_tuple):
    chunk_texts = list(chunk_texts_tuple)

    embeddings = encoder.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve(question, index, chunks):
    q_emb = encoder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, TOP_K)

    results = []

    for rank, idx in enumerate(ids[0]):
        score = float(scores[0][rank])

        if idx != -1 and score >= MIN_SCORE:
            chunk = chunks[idx]

            results.append({
                "rank": rank + 1,
                "score": score,
                "text": chunk["text"],
                "source": chunk.get("source", "Unknown Source"),
                "page": chunk.get("page", "N/A"),
                "type": chunk.get("type", "Unknown"),
                "chunk_id": chunk.get("chunk_id", "N/A")
            })

    return results


def get_answer_style_prompt(answer_style):
    if answer_style == "Brief":
        return """
Answer Style:
- Give a short and direct answer.
- Use 1-2 short paragraphs.
- Avoid unnecessary details.
"""

    if answer_style == "Detailed":
        return """
Answer Style:
- Give a detailed, student-friendly answer.
- Explain concepts step-by-step.
- Use bullet points where helpful.
- Include examples whenever relevant.
"""

    if answer_style == "Exam Notes":
        return """
Answer Style:
- Format the answer like exam notes.
- Use headings and bullet points.
- Include definition, key points, and example if available.
- Make it easy to revise.
"""

    return ""


def get_num_predict(answer_style):
    if answer_style == "Brief":
        return 300
    if answer_style == "Detailed":
        return 650
    if answer_style == "Exam Notes":
        return 600
    return 500


def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def ask_groq(question, context, source_mode, answer_style):
    api_key = get_groq_api_key()

    if not api_key:
        return (
            "GROQ_API_KEY is missing. Add it in Render Environment Variables "
            "or in Streamlit secrets."
        )

    style_prompt = get_answer_style_prompt(answer_style)
    max_tokens = get_num_predict(answer_style)

    system_prompt = f"""
You are a helpful AI teaching assistant.

Knowledge source selected by user: {source_mode}

Rules:
1. Use only the provided context.
2. Do not use outside knowledge unless clearly needed for a small explanation.
3. If the retrieved context is weak, unrelated, or does not clearly answer the question, say:
   "The selected knowledge source does not contain enough information about this."
4. Do not guess.
5. Keep the response clear and student-friendly.
6. When possible, naturally mention the source document or lecture context.

{style_prompt}
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Groq error: {e}"


def create_answer_pdf(question, answer, source_mode, answer_style):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    story = []

    safe_question = html.escape(question).replace("\n", "<br/>")
    safe_answer = html.escape(answer).replace("\n", "<br/>")
    safe_source = html.escape(source_mode)
    safe_style = html.escape(answer_style)

    story.append(Paragraph("Multi-Source RAG Knowledge Assistant", styles["Title"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Knowledge Source:</b> " + safe_source, styles["BodyText"]))
    story.append(Paragraph("<b>Answer Style:</b> " + safe_style, styles["BodyText"]))
    story.append(Paragraph("<b>LLM Model:</b> " + GROQ_MODEL, styles["BodyText"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Question:</b>", styles["Heading2"]))
    story.append(Paragraph(safe_question, styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Answer:</b>", styles["Heading2"]))
    story.append(Paragraph(safe_answer, styles["BodyText"]))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# -------------------- SIDEBAR --------------------
st.sidebar.markdown("""
<div class="sidebar-hero">
    <div class="sidebar-hero-title">⚙️ Settings</div>
    <div class="sidebar-hero-text">
        Configure your data source, answer style, documents, and model.
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-card">
    <div class="sidebar-card-title">📚 Knowledge Source</div>
    <div class="sidebar-card-text">Choose where the assistant should retrieve information from.</div>
</div>
""", unsafe_allow_html=True)

source_mode = st.sidebar.radio(
    "Knowledge Source",
    [
        "PDF Only",
        "Lecture Transcripts Only",
        "Both"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div class="sidebar-card">
    <div class="sidebar-card-title">✍️ Answer Format</div>
    <div class="sidebar-card-text">Select the response style.</div>
</div>
""", unsafe_allow_html=True)

answer_style = st.sidebar.selectbox(
    "Answer Style",
    [
        "Brief",
        "Detailed",
        "Exam Notes"
    ],
    index=1
)

st.sidebar.markdown("""
<div class="sidebar-card">
    <div class="sidebar-card-title">📄 Upload PDFs</div>
    <div class="sidebar-card-text">Upload one or more PDF documents.</div>
</div>
""", unsafe_allow_html=True)

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF notes/documents",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.sidebar.markdown(f"""
<div class="sidebar-card">
    <div class="sidebar-card-title">🤖 Active Model</div>
    <div class="sidebar-card-text">{GROQ_MODEL}</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")


# -------------------- SOURCE PROCESSING --------------------
all_chunks = []
pdf_count = 0
lecture_count = 0

if source_mode == "PDF Only":
    if uploaded_pdfs:
        with st.spinner("Reading uploaded PDFs..."):
            for pdf in uploaded_pdfs:
                pdf_chunks = extract_pdf_chunks(pdf)
                all_chunks.extend(pdf_chunks)

        pdf_count = len(uploaded_pdfs)
        st.sidebar.success(f"Processed {pdf_count} PDF(s)")
    else:
        st.markdown("""
        <div class="status-card">
            📄 Please upload one or more PDFs from the sidebar to begin.
        </div>
        """, unsafe_allow_html=True)

elif source_mode == "Lecture Transcripts Only":
    lecture_chunks = load_lecture_vectors()
    all_chunks.extend(lecture_chunks)
    lecture_count = len(lecture_chunks)

    if lecture_chunks:
        st.sidebar.success(f"Loaded {lecture_count} lecture chunks")
    else:
        st.error("vectors.joblib not found or could not be loaded.")

elif source_mode == "Both":
    lecture_chunks = load_lecture_vectors()
    all_chunks.extend(lecture_chunks)
    lecture_count = len(lecture_chunks)

    if uploaded_pdfs:
        with st.spinner("Reading uploaded PDFs..."):
            for pdf in uploaded_pdfs:
                pdf_chunks = extract_pdf_chunks(pdf)
                all_chunks.extend(pdf_chunks)

        pdf_count = len(uploaded_pdfs)
        st.sidebar.success(f"Combined: {pdf_count} PDF(s) + lecture transcripts")
    else:
        st.sidebar.warning("Upload PDFs to combine with lecture transcripts.")


# -------------------- MAIN APP --------------------
if all_chunks:
    document_names = sorted(
        list(set([c["source"] for c in all_chunks if c.get("type") == "PDF"]))
    )

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">PDF Documents</div>
            <div class="stat-value">{pdf_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Total Chunks</div>
            <div class="stat-value">{len(all_chunks)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Source Mode</div>
            <div class="stat-value">{source_mode}</div>
        </div>
        """, unsafe_allow_html=True)

    if document_names:
        st.caption("Uploaded PDFs: " + ", ".join(document_names))

    chunk_texts = [chunk["text"] for chunk in all_chunks]

    with st.spinner("Creating search index..."):
        index = cached_build_index(tuple(chunk_texts))

    st.markdown('<div class="ask-panel">', unsafe_allow_html=True)

    st.markdown("### Ask your question")

    input_mode = st.radio(
        "Choose input mode",
        ["Type Question", "Voice Question"],
        horizontal=True
    )

    question = ""

    if input_mode == "Type Question":
        question = st.text_input(
            "Type your question below:",
            placeholder="Example: Explain the main idea of these documents..."
        )

    else:
        st.info("Click Start Recording, speak your question, then click Stop Recording.")

        audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True
        )

        if audio:
            with st.spinner("Transcribing your voice question using Whisper..."):
                st.session_state.voice_question = transcribe_audio(audio["bytes"])

            st.success("Voice converted to text:")
            st.write(st.session_state.voice_question)

        question = st.text_input(
            "Transcribed question:",
            value=st.session_state.voice_question,
            placeholder="Your voice question will appear here..."
        )

    col_a, col_b = st.columns([1, 4])

    with col_a:
        get_answer = st.button("Get Answer")

    st.markdown('</div>', unsafe_allow_html=True)

    if get_answer:
        if question.strip():
            with st.spinner("Retrieving relevant context and generating answer..."):
                results = retrieve(question, index, all_chunks)

                if not results:
                    st.warning(
                        "No strong matching source found. Try asking more specifically or check the uploaded document."
                    )
                    st.stop()

                context_parts = []
                for r in results:
                    context_parts.append(
                        f"Source: {r['source']} | Page: {r['page']} | Type: {r['type']}\n{r['text']}"
                    )

                context = "\n\n".join(context_parts)
                answer = ask_groq(question, context, source_mode, answer_style)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "source_mode": source_mode,
                "answer_style": answer_style,
                "input_mode": input_mode
            })

            st.markdown("### Response")
            st.markdown(f"""
            <div class="answer-box">
                <span class="ai-label">✦ AI Assistant</span><br><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)

            pdf_bytes = create_answer_pdf(
                question,
                answer,
                source_mode,
                answer_style
            )

            st.download_button(
                label="📄 Download Answer as PDF",
                data=pdf_bytes,
                file_name="rag_answer.pdf",
                mime="application/pdf"
            )

            with st.expander("📌 View Sources Used"):
                for r in results:
                    source_label = "PDF Source" if r["type"] == "PDF" else "Lecture Source"
                    page_info = f"Page {r['page']}" if r["type"] == "PDF" else f"Chunk {r['chunk_id']}"

                    st.markdown(f"""
                    <div class="source-box">
                        <span class="source-badge">{source_label}</span><br>
                        <b>Source:</b> {r["source"]}<br>
                        <b>{page_info}</b><br>
                        <b>Similarity Score:</b> {r["score"]:.3f}
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Preview retrieved text - Source {r['rank']}"):
                        st.write(r["text"][:350])

        else:
            st.warning("Please enter or record a question first.")


# -------------------- CHAT HISTORY --------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 💬 Chat History")

    for i, item in enumerate(reversed(st.session_state.chat_history), 1):
        input_mode_text = item.get("input_mode", "Type Question")

        st.markdown(f"""
        <div class="history-box">
            <p><span class="question-label">Q{i}:</span> {item["question"]}</p>
            <p><span class="answer-label">Answer:</span> {item["answer"]}</p>
            <p style="font-size:13px;color:#475569;">
                Source: {item["source_mode"]} | Style: {item["answer_style"]} | Input: {input_mode_text}
            </p>
        </div>
        """, unsafe_allow_html=True)