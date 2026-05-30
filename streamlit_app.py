import os
import html
import tempfile
import json
import re
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

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "generated_quiz" not in st.session_state:
    st.session_state.generated_quiz = ""


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
        <span class="top-badge">🧠 Conversation Memory</span>
        <span class="top-badge">🎯 Quiz Generator</span>
        <span class="top-badge">📥 PDF Export</span>
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


def build_context(results):
    context_parts = []

    for r in results:
        context_parts.append(
            f"Source: {r['source']} | Page: {r['page']} | Type: {r['type']}\n{r['text']}"
        )

    return "\n\n".join(context_parts)


def build_conversation_memory():
    if not st.session_state.chat_history:
        return "No previous conversation."

    recent_history = st.session_state.chat_history[-3:]
    memory = []

    for item in recent_history:
        memory.append(
            f"Previous Question: {item['question']}\nPrevious Answer: {item['answer']}"
        )

    return "\n\n".join(memory)


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
        return 350
    if answer_style == "Detailed":
        return 800
    if answer_style == "Exam Notes":
        return 750
    return 600


def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def ask_groq(question, context, source_mode, answer_style):
    api_key = get_groq_api_key()

    if not api_key:
        return (
            "GROQ_API_KEY is missing. Add it in Streamlit secrets."
        )

    style_prompt = get_answer_style_prompt(answer_style)
    max_tokens = get_num_predict(answer_style)
    conversation_memory = build_conversation_memory()

    system_prompt = f"""
You are a helpful AI teaching assistant.

Knowledge source selected by user: {source_mode}

Rules:
1. Use only the provided retrieved context for factual answering.
2. Use conversation memory only to understand follow-up questions.
3. Do not invent facts.
4. If the retrieved context is weak, unrelated, or does not clearly answer the question, say:
   "The selected knowledge source does not contain enough information about this."
5. Keep the response clear and student-friendly.

{style_prompt}
"""

    user_prompt = f"""
Conversation Memory:
{conversation_memory}

Retrieved Context:
{context}

Current Question:
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


def generate_quiz(context, source_mode):
    api_key = get_groq_api_key()

    if not api_key:
        return []

    prompt = f"""
Create a clean MCQ quiz from the provided study material.

Return ONLY valid JSON. Do not include markdown, explanations outside JSON, or code fences.

JSON format:
[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "B",
    "explanation": "Short explanation"
  }}
]

Rules:
- Generate exactly 5 multiple choice questions.
- Each question must have exactly four options: A, B, C, D.
- The correct_answer value must be only A, B, C, or D.
- Use only the provided context.
- Keep it student-friendly.
- Do not invent facts.

Knowledge source: {source_mode}

Context:
{context}
"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You generate valid JSON MCQs from study material."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200
        )

        raw = response.choices[0].message.content.strip()

        # Remove possible markdown fences if the model adds them.
        raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        quiz_items = json.loads(raw)

        if not isinstance(quiz_items, list):
            return []

        return quiz_items[:5]

    except Exception as e:
        return [{
            "question": "Quiz generation failed",
            "options": {
                "A": "",
                "B": "",
                "C": "",
                "D": ""
            },
            "correct_answer": "",
            "explanation": str(e)
        }]


def render_quiz(quiz_items):
    st.markdown("### 🎯 Generated Quiz")

    for i, item in enumerate(quiz_items, 1):
        question = html.escape(item.get("question", ""))
        options = item.get("options", {})
        correct = html.escape(item.get("correct_answer", ""))
        explanation = html.escape(item.get("explanation", ""))

        option_a = html.escape(options.get("A", ""))
        option_b = html.escape(options.get("B", ""))
        option_c = html.escape(options.get("C", ""))
        option_d = html.escape(options.get("D", ""))

        st.markdown(
            f"""
            <div class="source-box" style="margin-bottom:24px; padding:24px;">
                <span class="source-badge">QUESTION {i}</span>

                <h3 style="margin-top:18px; margin-bottom:18px;">
                    {question}
                </h3>

                <p><b>A.</b> {option_a}</p>
                <p><b>B.</b> {option_b}</p>
                <p><b>C.</b> {option_c}</p>
                <p><b>D.</b> {option_d}</p>

                <div style="margin-top:18px; padding:14px; border-radius:14px; background:rgba(34,197,94,0.12);">
                    <b>Correct Answer:</b> {correct}
                </div>

                <div style="margin-top:14px;">
                    <b>Explanation:</b> {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


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


def create_chat_pdf(chat_history):
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

    story.append(Paragraph("Multi-Source RAG Knowledge Assistant - Chat History", styles["Title"]))
    story.append(Spacer(1, 16))

    for i, item in enumerate(chat_history, 1):
        question = html.escape(item["question"]).replace("\n", "<br/>")
        answer = html.escape(item["answer"]).replace("\n", "<br/>")
        source_mode = html.escape(item["source_mode"])
        answer_style = html.escape(item["answer_style"])
        input_mode = html.escape(item.get("input_mode", "Type Question"))

        story.append(Paragraph(f"<b>Q{i}:</b> {question}", styles["Heading2"]))
        story.append(Paragraph(f"<b>Answer:</b> {answer}", styles["BodyText"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"<b>Source:</b> {source_mode} | <b>Style:</b> {answer_style} | <b>Input:</b> {input_mode}",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 18))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def set_suggested_question(question):
    st.session_state.current_question = question
    st.session_state.voice_question = question


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
    ["PDF Only", "Lecture Transcripts Only", "Both"],
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
    ["Brief", "Detailed", "Exam Notes"],
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
    st.session_state.generated_quiz = ""
    st.sidebar.success("Chat history cleared.")


# -------------------- SOURCE PROCESSING --------------------
all_chunks = []
pdf_count = 0
lecture_count = 0

if source_mode == "PDF Only":
    if uploaded_pdfs:
        with st.spinner("Reading uploaded PDFs..."):
            for pdf in uploaded_pdfs:
                all_chunks.extend(extract_pdf_chunks(pdf))

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
                all_chunks.extend(extract_pdf_chunks(pdf))

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

    # -------------------- SUGGESTED QUESTIONS --------------------
    st.markdown("### 💡 Suggested Questions")

    suggested_questions = [
        "Summarize this document in simple language",
        "Explain the key concepts from this material",
        "Create exam notes from this PDF",
        "Generate 5 important questions from this document",
        "What are the most important points to remember?"
    ]

    q_cols = st.columns(3)

    for i, suggested_q in enumerate(suggested_questions):
        with q_cols[i % 3]:
            if st.button(suggested_q, key=f"suggested_{i}"):
                set_suggested_question(suggested_q)
                st.rerun()

    # -------------------- QUIZ GENERATOR --------------------
    st.markdown("### 🎯 Quiz Generator")

    if st.button("Generate MCQ Quiz"):
        with st.spinner("Generating quiz from your uploaded material..."):
            quiz_results = retrieve(
                "important concepts definitions examples exam questions quiz",
                index,
                all_chunks
            )

            if not quiz_results:
                fallback_chunks = all_chunks[:3]
                quiz_context = "\n\n".join([c["text"] for c in fallback_chunks])
            else:
                quiz_context = build_context(quiz_results)

            st.session_state.generated_quiz = generate_quiz(quiz_context, source_mode)

    if st.session_state.generated_quiz:
        render_quiz(st.session_state.generated_quiz)

    # -------------------- ASK PANEL --------------------
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
            value=st.session_state.current_question,
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

    get_answer = st.button("Get Answer")

    if get_answer:
        if question.strip():
            st.session_state.current_question = question

            with st.spinner("Retrieving relevant context and generating answer..."):
                results = retrieve(question, index, all_chunks)

                if not results:
                    st.warning(
                        "No strong matching source found. Try asking more specifically or check the uploaded document."
                    )
                    st.stop()

                context = build_context(results)
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

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                answer_pdf = create_answer_pdf(
                    question,
                    answer,
                    source_mode,
                    answer_style
                )

                st.download_button(
                    label="📄 Download This Answer",
                    data=answer_pdf,
                    file_name="rag_answer.pdf",
                    mime="application/pdf"
                )

            with col_d2:
                chat_pdf = create_chat_pdf(st.session_state.chat_history)

                st.download_button(
                    label="💬 Download Full Chat",
                    data=chat_pdf,
                    file_name="rag_chat_history.pdf",
                    mime="application/pdf"
                )

            st.markdown("### 📌 Sources Used")

            for r in results:
                source_label = "PDF Source" if r["type"] == "PDF" else "Lecture Source"
                page_info = f"Page {r['page']}" if r["type"] == "PDF" else f"Chunk {r['chunk_id']}"

                with st.expander(
                    f"{source_label} {r['rank']} | {r['source']} | {page_info} | Score {r['score']:.3f}"
                ):
                    st.markdown(f"""
                    <div class="source-box">
                        <span class="source-badge">{source_label}</span><br>
                        <b>Source:</b> {r["source"]}<br>
                        <b>{page_info}</b><br>
                        <b>Similarity Score:</b> {r["score"]:.3f}
                    </div>
                    """, unsafe_allow_html=True)

                    st.write(r["text"][:700])

        else:
            st.warning("Please enter or record a question first.")


# -------------------- CHAT HISTORY --------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 💬 Chat History with Memory")

    chat_pdf = create_chat_pdf(st.session_state.chat_history)

    st.download_button(
        label="💬 Download Complete Chat History",
        data=chat_pdf,
        file_name="rag_chat_history.pdf",
        mime="application/pdf"
    )

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