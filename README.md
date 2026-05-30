# 🚀 Multi-Source RAG Knowledge Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system that enables users to interact with PDFs, documents, and knowledge bases using both text and voice queries.

Built using Groq LLM, FAISS Vector Search, Whisper Speech Recognition, and Streamlit.

---

##  Features

### 📄 Document Intelligence
- Upload and process PDF documents
- Extract and analyze document content
- Semantic search across uploaded knowledge sources
- Context-aware question answering

### 🎤 Voice Assistant
- Voice-based question input
- Speech-to-text transcription using Whisper
- Hands-free interaction with documents

### 🤖 AI-Powered Responses
- Powered by Groq LLM
- Fast and accurate responses
- Context-aware answer generation
- Multiple answer styles

### 📚 Knowledge Retrieval
- FAISS vector database for semantic search
- Retrieval-Augmented Generation (RAG)
- Source-grounded responses
- Relevant context extraction

### 📝 Smart Learning Tools
- Automatic quiz generation
- Important question generation
- Summary generation
- Exam notes creation
- Key concept extraction

### 📥 Export Features
- Download generated answers as PDF
- Export study material
- Save AI-generated notes

---

## 🏗️ System Architecture

```text
User Query
     │
     ▼
PDF / Knowledge Base
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Search
     │
     ▼
Relevant Context Retrieval
     │
     ▼
Groq LLM
     │
     ▼
Generated Answer
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit
- HTML/CSS
- Custom UI Components

### AI & Machine Learning
- Groq API
- Llama 3.1
- OpenAI Whisper
- Sentence Transformers

### Vector Database
- FAISS

### Data Processing
- PyPDF
- NumPy
- Pandas

### Deployment
- Streamlit Community Cloud

---

## 📸 Key Functionalities

### PDF Question Answering
Ask questions directly from uploaded PDFs.

### Voice-Based Search
Use voice commands instead of typing.

### Quiz Generation
Generate MCQ quizzes automatically from uploaded material.

### Study Notes Generator
Convert large documents into concise exam notes.

### Smart Summarization
Generate easy-to-understand summaries from complex documents.

---

## 📂 Project Structure

```text
Multi-Source-RAG-Knowledge-Assistant/
│
├── streamlit_app.py
├── ask_question.py
├── search_transcripts.py
├── preprocess_json.py
├── process_incoming.py
├── requirements.txt
├── styles.css
├── vectors.joblib
│
├── jsons/
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/shrivastavasonal62-oss/Multi-Source-RAG-Knowledge-Assistant.git
```

### Navigate to Project

```bash
cd Multi-Source-RAG-Knowledge-Assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variable

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### Run Application

```bash
streamlit run streamlit_app.py
```

---

## 🚀 Live Demo

🔗 Add your deployed Streamlit link here

Example:

```text
https://your-app.streamlit.app
```

---

## 🎯 Use Cases

- AI Study Assistant
- Research Assistant
- Document Analysis
- Academic Learning
- Knowledge Retrieval
- Exam Preparation
- PDF Chatbot

---

## 📈 Future Enhancements

- Multi-PDF cross-document reasoning
- Chat history database
- User authentication
- Citation-based answers
- Image understanding
- OCR integration
- Agentic workflows

---

## 👨‍💻 Author

**Sonal Shrivastava**

B.Tech Computer Science Engineering

Interested in:
- Artificial Intelligence
- Generative AI
- Machine Learning
- Data Science
- Software Development

GitHub:
https://github.com/shrivastavasonal62-oss

LinkedIn:
(Add your LinkedIn URL)

---

## ⭐ If you like this project

Give this repository a star ⭐ and support the project.
