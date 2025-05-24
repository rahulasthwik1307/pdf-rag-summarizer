
## 📄 PDF RAG Summarizer with Groq Models

An AI-powered PDF question-answering app using **Streamlit**, **FAISS**, **Sentence Transformers**, and **Groq LLMs**. Upload any PDF, ask questions about its content, and get intelligent responses with references to the original pages.

---

## 🚀 Features

* ✅ Upload and read any PDF
* ✅ View all pages visually
* ✅ Ask questions about the document
* ✅ Use **Groq LLMs**: LLaMA 4 Maverick, Deepseek 70B, LLaMA 3 70B
* ✅ RAG-based context retrieval using FAISS
* ✅ Choose primary and secondary models for comparison
* ✅ Parallel dual-model comparison with side-by-side layout
* ✅ Intelligent `<think>` rendering for Deepseek
* ✅ Shows source page numbers for every answer
* ✅ Beautiful and clean interface with spinners and bubbles
* ✅ Optional: Suppress terminal warnings/errors for cleaner logs

---

## 🧠 Tech Stack

* **Python**
* **Streamlit**
* **Groq API**
* **FAISS**
* **Sentence Transformers**
* **PyPDF2**
* **PIL**
* **pdf2image**

---

## 🗃️ Folder Structure

```
pdf-rag-summarizer/
│
├── app.py                   # Streamlit main application
├── requirements.txt         # Project dependencies
├── .env                     # Contains your GROQ_API_KEY
│
├── utils/                   # Utility functions
│   ├── pdf_processor.py     # Extract and chunk PDF text
│   └── rag_handler.py       # FAISS index management
│
├── data/                    # Saved indexes and metadata
│   └── <pdf_name>_faiss/
│       ├── index.faiss
│       ├── chunks.pkl
│       └── chunk_to_page_map.pkl
└── ...
```

---

## 📚 How It Works

1. **PDF Upload**
   User uploads a PDF file. It is immediately parsed page-by-page using `PyPDF2`.

2. **Text Chunking**
   Each page is split into overlapping chunks (e.g., 1000 characters with 200 character overlap) for better semantic continuity using `chunk_text()`.

3. **Embedding & Indexing**
   The chunks are converted into vector embeddings using `sentence-transformers` and stored in a **FAISS** index along with the original chunk-to-page mapping.

4. **User Question**
   User types a question into the UI. This question is embedded and matched against the FAISS index to retrieve the most relevant text chunks.

5. **Response Generation**
   The retrieved chunks are passed to the selected **Groq LLM** model (e.g., Llama 4, Deepseek 70B) as context. The model answers only based on the provided context.

6. **Dual Model Comparison (Optional)**
   If selected, the question is sent to a second model in parallel and results are displayed side-by-side for comparison with token difference metrics.

7. **Source Transparency**
   The model’s response includes references to the pages from which the information was retrieved.

8. **Visual Output**
   Clean, styled answers are shown with optional Deepseek `<think>` reasoning and contextual bubble UI.

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/rahulasthwik1307/pdf-rag-summarizer.git
cd pdf-rag-summarizer
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variable

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💻 Running the App

### ▶️ Run Normally

```bash
streamlit run app.py
```

### 🧹 Run with Clean Terminal (Suppress Errors)

1. At the top of `app.py`, add:

```python
import warnings
warnings.filterwarnings("ignore")
```

2. Then run:

```bash
streamlit run app.py --logger.level=error --server.runOnSave false
```

---

## 📦 Requirements

```
streamlit
groq
PyPDF2
pdf2image
Pillow
faiss-cpu
numpy
sentence-transformers
```

> If you're using GPU, replace `faiss-cpu` with `faiss-gpu`.

---

## 🌟 Special Notes

* Fully supports **Deepseek-style thinking logs**
* Fast and interactive document-based reasoning


