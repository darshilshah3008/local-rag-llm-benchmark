
# NotebookLM-style Local RAG with Mistral + BGE-M3

A fully local, privacy-friendly **Retrieval-Augmented Generation (RAG)** system inspired by Google's NotebookLM. Chat with your PDF documents using **Ollama** LLMs and **BGE-M3** embeddings — everything runs on your machine, no data leaves your computer.

## Purpose

This repo is designed to **compare two LLMs of different sizes** running the same RAG pipeline with the **same embedding model (BGE-M3)**. The goal is to evaluate:

- **Generation speed** — how fast each model produces answers.
- **Output quality** — how accurate and useful the answers are.

By keeping the retrieval layer identical (BGE-M3 + FAISS) and only swapping the answering LLM, you get a fair side-by-side comparison. For example:

| Component        | Model              |
|------------------|-------------------|
| Embeddings       | `bge-m3`          |
| LLM #1           | `mistral`         |
| LLM #2           | `gemma-manual`    |

Switch between models by changing one environment variable (`OLLAMA_MODEL`) — no re-ingestion or re-embedding needed.

## What This Repo Does

1. **Ingest PDFs** — extract text (with OCR fallback for scanned pages) and tables from PDF files.
2. **Build a vector index** — chunk the extracted text NotebookLM-style, embed chunks with BGE-M3, and store them in a FAISS index.
3. **Ask questions** — query your documents in natural language; the system retrieves the most relevant chunks and generates answers using a local LLM of your choice.
4. **Compare LLMs** — run the same questions against different models and compare speed and quality.

The entire pipeline runs offline — no API keys, no cloud services, no data upload.

## Project Structure

```
├── data/
│   ├── pdfs/            # Drop your PDF files here
│   └── index/           # Auto-generated FAISS index + metadata
│
├── ingest.py            # PDF text/table extraction, OCR fallback, chunking
├── embedding.py         # Embed chunks via Ollama BGE-M3, build FAISS index
├── search.py            # Semantic retriever (FAISS cosine similarity search)
├── rag.py               # Main entry point — ingest & interactive Q&A loop
├── utils.py             # Ollama connection config, chat/embedding helpers
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

- **Python 3.10+**
- **Ollama** — https://ollama.com/download
- **16 GB RAM** minimum (32–48 GB recommended for larger models)
- **(Optional) Tesseract OCR** — only needed if your PDFs contain scanned/image pages
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/darshilshah3008/local-rag-llm-benchmark
cd local-rag-llm-benchmark
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Pull the Ollama models

You need the **embedding model** plus **both LLMs** you want to compare.

```bash
# Embedding model
ollama pull bge-m3

# LLMs
ollama pull mistral

# If using custom Gemma model
ollama create gemma-manual -f Modelfile
```

### 4. Add your PDFs

Place one or more PDF files into the `data/pdfs/` folder.

### 5. Build the FAISS index

```bash
python rag.py --ingest
```

This extracts text and tables from every PDF, chunks the content, generates embeddings via BGE-M3, and saves a FAISS index to `data/index/`.

> Re-run this command whenever you add or change PDFs.

### 6. Start chatting

```bash
python rag.py
```

```
RAG is ready. Ask questions about your PDFs.
Type 'exit' or 'quit' to stop.

Q> What is this PDF about?
Q> Summarize the key points on page 3.
Q> What are the main deliverables?
Q> exit
```

## Comparing Two LLMs

The embedding index only needs to be built **once**. After that, swap the answering LLM without re-ingesting:

On **Windows PowerShell**, set the variable first:

```powershell
# Run with Mistral (~7B)
$env:OLLAMA_MODEL="mistral"
$env:EMBEDDING_MODEL="bge-m3"
python rag.py

# Run with Gemma (~2B)
$env:OLLAMA_MODEL="gemma-manual"
$env:EMBEDDING_MODEL="bge-m3"
python rag.py
```

Or create a `.env` file in the project root:

```dotenv
OLLAMA_MODEL=mistral   # change to gemma-manual for the smaller model
```

Ask the **same questions** with each model and compare:

| Metric | Mistral (~7B) | Qwen 2.5 3B (~3B) |
|---|---|---|
| Response time | Slower | Faster |
| Answer depth | More detailed | More concise |
| RAM usage | Higher | Lower |
| Accuracy | Baseline | Compare against baseline |

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | LLM used for answering |
| `EMBEDDING_MODEL` | `bge-m3` | Model used for text embeddings |
| `PDF_DIR` | `data/pdfs` | Folder containing input PDFs |
| `INDEX_DIR` | `data/index` | Folder for the FAISS index and metadata |

## How It Works

```
PDF files
  │
  ▼
ingest.py ── extract text (pdfplumber) + OCR fallback (pytesseract)
           ── extract tables (camelot)
           ── chunk text (~1200 chars, paragraph-aware)
  │
  ▼
embedding.py ── embed chunks via Ollama BGE-M3
             ── normalize vectors
             ── build FAISS IndexFlatIP (cosine similarity)
             ── save index + metadata to disk
  │
  ▼
rag.py ── user asks a question
       ── search.py retrieves top-k similar chunks (FAISS)
       ── builds a context block with file names & page numbers
       ── sends context + question to Mistral via Ollama
       ── prints the answer
```

## Troubleshooting

| Problem | Solution |
|---|---|
| **TimeoutError** from Ollama | Switch to a smaller model: `ollama pull qwen2.5:3b-instruct` and set `OLLAMA_MODEL=qwen2.5:3b-instruct` |
| **Slow answers** | Use a lighter model like `qwen2.5:3b-instruct` |
| **Want better quality** | Use `qwen2.5:7b-instruct` or `llama3.1:8b-instruct` (needs more RAM) |
| **OCR not working** | Install Tesseract and make sure `tesseract` is on your PATH |
| **FAISS index missing** | Run `python rag.py --ingest` before asking questions |

## License

This project is provided for educational and personal use.
