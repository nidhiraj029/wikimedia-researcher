# Wikimedia RAG Researcher

A Retrieval-Augmented Generation (RAG) research assistant built with Python for searching and retrieving relevant information from Wikimedia-based datasets.

The project processes documents, splits them into meaningful chunks, generates vector embeddings, stores them locally, and uses similarity search to retrieve the most relevant information for a research query.

## 🚀 Features

* 📚 **Document ingestion** — Processes Wikimedia research data.
* ✂️ **Text chunking** — Splits large documents into smaller searchable chunks.
* 🧠 **Semantic embeddings** — Converts text into numerical vector representations.
* 🔎 **Vector similarity search** — Retrieves relevant information using embeddings.
* ⚡ **FAISS-based retrieval** — Enables fast similarity searching over thousands of vectors.
* 💾 **Metadata management** — Maintains metadata associated with indexed content.
* 🤖 **RAG pipeline** — Combines retrieval with language-model-based research workflows.
* 🌐 **Python application interface** — Provides an application entry point through `app.py`.

## 🏗️ Project Architecture

```text
                     ┌──────────────────────┐
                     │   Wikimedia Dataset  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │     Data Ingestion   │
                     │    ingestion.py      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │     Text Chunking    │
                     │      chunker.py      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Embedding Creation  │
                     │    embeddings.py     │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │    Vector Storage    │
                     │   Local Embeddings   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Semantic Retrieval │
                     │    retriever.py      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │      RAG Engine      │
                     │       rag.py         │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │    Research Answer   │
                     └──────────────────────┘
```

## 📁 Project Structure

```text
wikimedia-researcher/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── inspect_dataset.py
│   ├── rag.py
│   └── retriever.py
│
└── embeddings/
    ├── embeddings.npy
    └── metadata.json
```

> The `embeddings/` directory contains generated local artifacts and is excluded from Git tracking.

## 🔧 Technologies Used

* **Python**
* **FAISS** — Vector similarity search
* **NumPy** — Numerical and vector operations
* **Sentence Transformers / Hugging Face** — Text embeddings
* **JSON** — Metadata storage
* **Git & GitHub** — Version control

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/nidhiraj029/wikimedia-researcher.git
cd wikimedia-researcher
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

The RAG pipeline can be executed with:

```bash
python src/rag.py
```

The application entry point can be run with:

```bash
python app.py
```

## 🔄 RAG Pipeline

The system follows these major stages:

### 1. Data Ingestion

Raw Wikimedia data is loaded and processed using the ingestion pipeline.

### 2. Chunking

Large documents are divided into smaller chunks to improve retrieval accuracy and context management.

### 3. Embedding Generation

Each text chunk is converted into a numerical vector using an embedding model.

### 4. Vector Storage

Generated embeddings are stored locally and used for similarity search.

### 5. Retrieval

FAISS searches the vector space and identifies the chunks that are most semantically relevant to a user's query.

### 6. RAG Processing

The retrieved context is passed into the RAG pipeline to generate a research-oriented response.

## 📊 Current Dataset / Index

The current local index contains approximately:

```text
26,234 vectors
```

The generated embedding files are intentionally excluded from GitHub through `.gitignore`.

This keeps the repository lightweight and prevents large generated artifacts from being committed.

## 🧪 Example

After starting the RAG pipeline:

```bash
python src/rag.py
```

The system loads:

```text
WIKIMEDIA RAG RESEARCHER
Loading retriever...
Loading FAISS index...
Loading embedding model...
```

A research query can then be processed against the indexed Wikimedia content.

## 🔐 Git & Data Management

Generated data and machine-specific files are excluded using `.gitignore`.

Examples include:

```gitignore
__pycache__/
venv/
.env
embeddings/
*.pkl
*.pickle
*.joblib
```

This prevents unnecessary or potentially sensitive local files from being uploaded to the repository.

## 🛣️ Future Improvements

Planned improvements include:

* [ ] Improve retrieval accuracy
* [ ] Add advanced query rewriting
* [ ] Add hybrid keyword + semantic search
* [ ] Improve citation and source tracking
* [ ] Add conversation history
* [ ] Add a web-based research interface
* [ ] Add automated document updating
* [ ] Improve RAG answer generation
* [ ] Add evaluation metrics for retrieval quality
* [ ] Deploy the application

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Make your changes.
4. Commit your changes.

```bash
git add .
git commit -m "Add your feature"
```

5. Push the branch.

```bash
git push origin feature/your-feature
```

6. Open a Pull Request.

## 📄 License

This project currently does not specify a license.

If you plan to make the project open source, consider adding an appropriate license such as MIT.

## 👨‍💻 Author

**Nidhiraj Bhandari**

GitHub:
https://github.com/nidhiraj029

---

⭐ If you find this project useful, consider giving the repository a star.
