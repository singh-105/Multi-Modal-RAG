# EuroJobsCenter Germany Handbook — Voice & Text RAG Assistant

An English-only, policy-grounded multimodal Voice & Text Retrieval-Augmented Generation (RAG) assistant designed for the official **EuroJobsCenter Visa & Immigration Handbook for Germany**.

The assistant allows international applicants, professional drivers, and skilled workers to query German visa regulations, EU Blue Cards, Code 95 qualifications, driver declarations, and residency procedures using natural voice conversation or text chat, with real-time audio playback and strict factual citation.

---

## 🛠️ Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Backend Web Framework** | Flask 3.x | Lightweight REST API server with session memory and audio endpoints |
| **Language Model (LLM)** | EuroLLM-1.7B-Instruct | 1.7-billion parameter instruction-tuned causal language model |
| **Embedding Model** | multilingual-e5-base | 768-dimensional multilingual text embedding model |
| **Vector Database** | Qdrant | High-performance embedded vector search engine |
| **Speech-to-Text (STT)** | Whisper-base.en | End-to-end English automatic speech recognition |
| **Text-to-Speech (TTS)** | Piper Neural TTS | Ultra-fast local neural speech synthesizer via ONNX Runtime |
| **Frontend** | Vanilla JS / CSS3 / HTML5 | Dark-slate UI (`#0b0f19` / `#10b981`), Web Audio API, sentence chunk prefetch |

---

## 🏗️ System Architecture

```
User Voice Input (Microphone)
             │
             ▼
   [Web Audio API (16kHz WAV)]
             │
             ▼
    Whisper STT Model
             │ (Transcribed Query)
             ▼
   Multilingual E5 Embeddings
             │
             ▼
   Qdrant Vector DB (Cosine Similarity)
             │
             ▼
   Hybrid Reranker & Factual Grounding Engine (grounding.py)
   ├── Section Diversity (max 2/section)
   ├── Lexical IDF Boosting
   └── Out-of-Scope Detection
             │
             ▼
   EuroLLM-1.7B-Instruct (Grounded Prompting)
             │
             ▼
   Factual Post-Verification Engine
   ├── Numeric & Cost Sanitization
   └── Guarantee Disclaimer Enforcement
             │
             ▼
   Piper Neural TTS (Sentence-Chunked Audio Streaming)
             │
             ▼
Browser Audio Playback + Formatted Handbook Citations
```

---

## 📁 Repository Structure

```text
├── app.py                      # Main Flask application & voice/text API endpoints
├── grounding.py                # Hybrid retrieval, grounding & factual verification engine
├── ingest_qdrant.py            # Knowledge base ingestion script for Qdrant
├── requirements.txt            # Production Python package dependencies
├── .env.example                # Configuration template for environment variables
├── .gitignore                  # Git exclusion rules for virtualenvs, caches & models
│
├── static/                     # Frontend web client assets
│   ├── app.js                  # Audio recording, playback queue & chat interaction
│   └── style.css               # EuroJobsCenter responsive dark-theme design
│
├── templates/
│   └── index.html              # Main chat interface with bottom-bar mic control
│
├── output/                     # 10 verified handbook chunk files (102 JSONL chunks)
│   ├── before_you_begin.jsonl
│   ├── changing_employers.jsonl
│   ├── code95_driver_card.jsonl
│   ├── driver_declaration.jsonl
│   ├── faq.jsonl
│   ├── first_days_germany.jsonl
│   ├── first_page.jsonl
│   ├── important_security_notice.jsonl
│   ├── your_career_germany.jsonl
│   └── your_journey_to_germany.jsonl
│
├── before_you_begin.py         # Chunk extractor for "Before You Begin" section
├── changing_employers.py       # Chunk extractor for "Changing Employers" section
├── code95.py                   # Chunk extractor for "Code 95 & Driver Card" section
├── driver_declaration.py       # Chunk extractor for "Driver Declaration" section
├── faq.py                      # Chunk extractor for "FAQ" section
├── first_days_germany.py       # Chunk extractor for "First Days in Germany" section
├── first_page.py               # Chunk extractor for "Handbook Overview" section
├── important_security_notice.py# Chunk extractor for "Important Security Notice" section
├── your_career_germany.py      # Chunk extractor for "Your Career in Germany" section
├── your_journey_to_germany.py  # Chunk extractor for "Your Journey to Germany" section
│
├── test_voice_agent.py         # 16-test API, STT, and voice regression suite
├── test_grounding_suite.py     # 7-test factual grounding & hallucination test suite
└── test_search.py              # Semantic retrieval smoke test
```

---

## 💻 System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Python 3.10, 3.11, or 3.12
- **Memory (RAM):** Minimum 8 GB (16 GB recommended)
- **Disk Space:** ~5 GB free space for model weights cache (`intfloat/multilingual-e5-base`, `utter-project/EuroLLM-1.7B-Instruct`, `openai/whisper-base.en`, and Piper voice)
- **Hardware Acceleration:**
  - **NVIDIA GPU (CUDA):** Recommended for sub-second generation (requires 6 GB+ VRAM).
  - **CPU Only:** Fully supported with automatic fallback to `torch.float32`.

---

## 🚀 Setup & Installation (Windows)

### 1. Clone the Repository

```powershell
git clone https://github.com/veloitsolutions/eurojobscenter-agent.git
cd eurojobscenter-agent
```

### 2. Create and Activate a Python Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment in PowerShell
.\.venv\Scripts\activate

# If PowerShell script execution is restricted, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file from the provided template:

```powershell
Copy-Item .env.example .env
```

*(Default settings run on `127.0.0.1:7860` with automatic CPU/GPU detection).*

---

## 📚 Knowledge Base Ingestion

Before starting the assistant for the first time, ingest the 10 handbook chunk files into the local Qdrant vector database:

```powershell
python ingest_qdrant.py
```

Expected output:
```text
Loading multilingual-e5-base embedder...
Ingesting 10 chunk files into Qdrant...
Ingested 102 chunks into collection 'veloit_handbook_chunks'.
Ingestion complete. Vector DB is ready.
```

---

## 🎙️ Running the Application

### 1. Start the Flask Backend

```powershell
python app.py
```

The server will automatically:
1. Detect compute hardware (CUDA GPU or CPU fallback).
2. Connect to the local embedded Qdrant storage (`qdrant_data/`).
3. Initialize the Factual Grounding Engine over all indexed points.
4. Load EuroLLM tokenizer and model into memory.
5. Warm up Piper Neural TTS in a background thread.
6. Start serving at `http://127.0.0.1:7860`.

### 2. Open the Frontend

Open your web browser and navigate to:
```text
http://127.0.0.1:7860
```

---

## 🗣️ Using the Assistant

### Text Chat
- Type your question into the chat input bar and press **Enter** or click **Send**.
- Citations and verified handbook section headers are displayed with the answer.

### Voice Mode & Microphone Input
1. Allow browser microphone access when prompted.
2. Click the **Microphone** icon in the bottom composer bar (turns green when active).
3. Speak your question clearly (e.g. *"What documents do I need for the driver declaration?"*).
4. When you stop speaking, the assistant automatically detects silence, transcribes your speech with Whisper, retrieves verified handbook facts, synthesizes the answer with Piper TTS, and speaks it aloud through your speakers.
5. To toggle Voice Mode off, click the Microphone button again.

---

## 🎧 Audio Dependencies & Models (Whisper, Piper, FFmpeg)

- **Whisper STT (`openai/whisper-base.en`):** Automatically downloaded via Hugging Face Transformers into `models/`. Audio is recorded directly via Web Audio API at 16,000 Hz mono PCM WAV format.
- **Piper Neural TTS (`rhasspy/piper-voices`):** `en_US-lessac-medium.onnx` and its configuration JSON are downloaded automatically on first startup into `models/`. Runs via ONNX Runtime on CPU for fast sentence chunk synthesis.
- **FFmpeg:** Modern browsers send standard PCM WAV data directly to the `/api/voice/transcribe` endpoint, so FFmpeg is not strictly required for browser audio. However, having FFmpeg installed in your system PATH is recommended if processing external compressed media formats.

---

## 🧪 Running the Verification Test Suites

The project includes 24 automated unit and integration tests verifying retrieval, factual grounding, and voice endpoints.

### 1. Factual Grounding Test Suite (7 Tests)
Verifies out-of-scope refusals, Code 95 cost hallucination prevention, and guarantee disclaimer enforcement:

```powershell
python -m unittest test_grounding_suite.py
```

### 2. Voice Agent & API Suite (16 Tests)
Verifies `/health`, `/api/voice/status`, `/reset`, `/ask`, audio transcription, and sentence-chunked TTS:

```powershell
python -m unittest test_voice_agent.py
```

### 3. Retrieval Smoke Test
Verifies cosine similarity search directly against Qdrant:

```powershell
python test_search.py
```

### 4. Combined Execution
```powershell
python -m unittest test_grounding_suite.py test_voice_agent.py test_search.py
```

